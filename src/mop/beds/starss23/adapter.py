"""Component 1b: the STARSS23 data adapter.

This module parses the native STARSS23 / DCASE onset-metadata schema and serves the frozen ``Clip``
contract of ``schema.py``. It is the single seam between raw corpus media and the scored bed: the same
parse path handles synthetic fixtures now and real STARSS23 files later, unchanged.

Two adapters implement one protocol:

- ``SyntheticStarssAdapter`` wraps deterministic in-memory (or on-disk) fixtures. Its ``source_kind`` is
  ``"synthetic"``, so the producer downstream refuses any verdict above a mechanics demonstration.
- ``RealStarssAdapter`` is a blocked stub. It carries the identical parse wiring so a real, MIT-licensed
  STARSS23 tree slots in unchanged, but it refuses to load in this no-download workflow. Only its audio
  decode is deferred; the metadata, filename, and onset-derivation paths are shared and exercised now.

The STARSS23 metadata grid is the DCASE onset contract: one row per active 100 ms frame per source,
carrying ``frame, class, source, azimuth, elevation`` and, from STARSS23 on, ``distance`` (integer cm).
An onset is the first frame of a contiguous activity run for a ``(class, source)`` track. Because the
referee scores onset presence per frame, simultaneous onsets collapse to one label per frame, matching
the ``Clip.onsets`` at-most-one-per-frame contract.

Recomputing ``audio_sha256`` is raw transport work, charged to ``WorkVector.raw_transport_and_adapters``
and kept off the matched arm budget.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mop.escs.accounting import WorkVector

from .schema import (
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    Clip,
    ClipSplit,
    OnsetEvent,
    room_disjoint_split,
)

# The two provenance-bearing source kinds. Synthetic can never earn more than a mechanics verdict.
SOURCE_KIND_SYNTHETIC = "synthetic"
SOURCE_KIND_REAL = "real"

# STARSS23 dev-set folds. Fold 3 is the room-disjoint dev-train (90 clips), fold 4 the dev-test (78).
FOLD_DEV_TRAIN = 3
FOLD_DEV_TEST = 4

# A distance column may be absent in pre-2023 STARSS metadata. The bed requires it, so a row that lacks
# it parses with this sentinel and is refused when an onset is built from it.
DISTANCE_ABSENT = -1
# STARSS23 azimuth and elevation are integer degrees; distance is integer centimeters.
_AZIMUTH_MIN, _AZIMUTH_MAX = -180, 180
_ELEVATION_MIN, _ELEVATION_MAX = -90, 90
_DISTANCE_MAX_CM = 100_000  # 1 km ceiling; a physical sanity bound, not a knob

_CLIP_NAME_RE = re.compile(r"^fold(\d+)_room(\d+)_mix(\d+)$")


class AdapterRefusal(ValueError):
    """Raised when corpus media or metadata violates the STARSS23 adapter contract."""


class RealDataBlocked(AdapterRefusal):
    """Raised when the real STARSS23 adapter is used in this no-download workflow."""


# ---------------------------------------------------------------------------
# Clip filename identity: fold, room, mix.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipName:
    """The parsed STARSS23 clip filename ``fold{F}_room{R}_mix{M}``."""

    fold: int
    room: int
    mix: int

    def __post_init__(self) -> None:
        for name, value in (("fold", self.fold), ("room", self.room), ("mix", self.mix)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AdapterRefusal(f"ClipName.{name} must be a nonnegative integer")

    @property
    def clip_id(self) -> str:
        return f"fold{self.fold}_room{self.room}_mix{self.mix:03d}"

    @property
    def room_id(self) -> str:
        # Zero-padded so lexical room order matches numeric order for the schema's sorted room split.
        # STARSS23 has far fewer than 100 rooms, so two digits keep fold-3 rooms sorted before fold-4.
        return f"room{self.room:02d}"


def parse_clip_name(name: str) -> ClipName:
    """Parse a STARSS23 clip id or filename (any audio/metadata extension is stripped)."""

    if not isinstance(name, str) or not name.strip():
        raise AdapterRefusal("clip name must be a nonempty string")
    stem = Path(name.strip()).stem if "." in name else name.strip()
    match = _CLIP_NAME_RE.fullmatch(stem)
    if match is None:
        raise AdapterRefusal(f"clip name {name!r} is not fold<F>_room<R>_mix<M>")
    return ClipName(fold=int(match.group(1)), room=int(match.group(2)), mix=int(match.group(3)))


def format_clip_id(fold: int, room: int, mix: int) -> str:
    return ClipName(fold=fold, room=room, mix=mix).clip_id


# ---------------------------------------------------------------------------
# Native DCASE / STARSS23 onset-metadata rows.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetadataRow:
    """One native STARSS23 metadata row: an active 100 ms frame of one source with its DOA and distance."""

    frame: int
    class_id: int
    source_id: int
    azimuth: int
    elevation: int
    distance: int = DISTANCE_ABSENT

    def __post_init__(self) -> None:
        for name, value in (
            ("frame", self.frame),
            ("class_id", self.class_id),
            ("source_id", self.source_id),
            ("azimuth", self.azimuth),
            ("elevation", self.elevation),
            ("distance", self.distance),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdapterRefusal(f"MetadataRow.{name} must be an integer")
        if self.frame < 0:
            raise AdapterRefusal("MetadataRow.frame must be nonnegative")
        if not 0 <= self.class_id < N_CLASSES:
            raise AdapterRefusal(f"MetadataRow.class_id must be in [0, {N_CLASSES})")
        if self.source_id < 0:
            raise AdapterRefusal("MetadataRow.source_id must be nonnegative")
        if not _AZIMUTH_MIN <= self.azimuth <= _AZIMUTH_MAX:
            raise AdapterRefusal("MetadataRow.azimuth must lie in [-180, 180] degrees")
        if not _ELEVATION_MIN <= self.elevation <= _ELEVATION_MAX:
            raise AdapterRefusal("MetadataRow.elevation must lie in [-90, 90] degrees")
        if self.distance != DISTANCE_ABSENT and not 0 < self.distance <= _DISTANCE_MAX_CM:
            raise AdapterRefusal("MetadataRow.distance must be a positive centimeter count")

    @property
    def has_distance(self) -> bool:
        return self.distance != DISTANCE_ABSENT

    def sort_key(self) -> tuple[int, int, int]:
        return (self.frame, self.class_id, self.source_id)


def parse_starss23_metadata(text: str) -> tuple[MetadataRow, ...]:
    """Parse native STARSS23 metadata text into rows.

    Accepts the six-column STARSS23 form ``frame,class,source,azimuth,elevation,distance`` and the
    five-column pre-2023 form without distance (distance then parses as ``DISTANCE_ABSENT``). Blank
    lines are ignored. This is the exact seam a real STARSS23 file passes through unchanged.
    """

    if not isinstance(text, str):
        raise AdapterRefusal("metadata text must be a string")
    rows: list[MetadataRow] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in (5, 6):
            raise AdapterRefusal(f"metadata line {lineno} must have 5 or 6 columns, saw {len(parts)}")
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise AdapterRefusal(f"metadata line {lineno} has a non-integer field") from exc
        if len(values) == 5:
            frame, class_id, source_id, azimuth, elevation = values
            distance = DISTANCE_ABSENT
        else:
            frame, class_id, source_id, azimuth, elevation, distance = values
        rows.append(
            MetadataRow(
                frame=frame,
                class_id=class_id,
                source_id=source_id,
                azimuth=azimuth,
                elevation=elevation,
                distance=distance,
            )
        )
    return tuple(rows)


def format_starss23_metadata(rows: Sequence[MetadataRow]) -> str:
    """Serialize rows to native STARSS23 metadata text, sorted by ``(frame, class, source)``.

    Six columns are written when every row carries a distance, five when none do. A mixed set is
    refused: a partial distance column is not a valid STARSS23 file.
    """

    rows = tuple(rows)
    if not rows:
        return ""
    with_distance = sum(1 for row in rows if row.has_distance)
    if with_distance not in (0, len(rows)):
        raise AdapterRefusal("cannot serialize a metadata set with only some distance labels")
    include_distance = with_distance == len(rows)
    lines: list[str] = []
    for row in sorted(rows, key=MetadataRow.sort_key):
        fields = [row.frame, row.class_id, row.source_id, row.azimuth, row.elevation]
        if include_distance:
            fields.append(row.distance)
        lines.append(",".join(str(value) for value in fields))
    return "\n".join(lines) + "\n"


def _prefer_onset(candidate: OnsetEvent, incumbent: OnsetEvent) -> bool:
    """Deterministic tie-break at a co-onset frame: keep the nearest source, then a stable order."""

    return (candidate.distance, candidate.class_id, candidate.azimuth, candidate.elevation) < (
        incumbent.distance,
        incumbent.class_id,
        incumbent.azimuth,
        incumbent.elevation,
    )


def onset_events_from_rows(rows: Iterable[MetadataRow]) -> tuple[OnsetEvent, ...]:
    """Derive onset labels from dense per-frame activity.

    An onset is the first frame of a contiguous activity run for a ``(class, source)`` track. Because
    the referee scores onset presence per frame, simultaneous onsets from different sources collapse to
    a single label per frame (the nearest source wins, with a stable tie-break). A row missing its
    distance label is refused: the bed requires STARSS23 distance.
    """

    by_track: dict[tuple[int, int], list[MetadataRow]] = {}
    for row in rows:
        if not row.has_distance:
            raise AdapterRefusal("onset derivation requires STARSS23 distance labels")
        by_track.setdefault((row.class_id, row.source_id), []).append(row)
    onset_by_frame: dict[int, OnsetEvent] = {}
    for track_rows in by_track.values():
        track_rows.sort(key=lambda row: row.frame)
        previous: int | None = None
        for row in track_rows:
            if previous is None or row.frame != previous + 1:
                event = OnsetEvent(
                    frame=row.frame,
                    class_id=row.class_id,
                    azimuth=float(row.azimuth),
                    elevation=float(row.elevation),
                    distance=float(row.distance),
                )
                incumbent = onset_by_frame.get(row.frame)
                if incumbent is None or _prefer_onset(event, incumbent):
                    onset_by_frame[row.frame] = event
            previous = row.frame
    return tuple(sorted(onset_by_frame.values(), key=lambda event: event.frame))


def _as_int_label(value: float, label: str) -> int:
    """Coerce an integer-valued onset field to int. Native STARSS23 metadata is integer only."""

    number = float(value)
    rounded = round(number)
    if abs(number - rounded) > 1e-9:
        raise AdapterRefusal(f"onset {label} {value!r} is not integer-valued, cannot serialize to STARSS23")
    return int(rounded)


def metadata_rows_from_onsets(
    onsets: Iterable[OnsetEvent], *, active_frames: int = 1
) -> tuple[MetadataRow, ...]:
    """Serialize onset labels into dense native STARSS23 rows, one source per onset.

    Each onset becomes ``active_frames`` consecutive active rows for its own source track, exactly the
    dense form ``onset_events_from_rows`` inverts. Requires integer-valued degrees and centimeters.
    """

    if isinstance(active_frames, bool) or not isinstance(active_frames, int) or active_frames < 1:
        raise AdapterRefusal("active_frames must be a positive integer")
    rows: list[MetadataRow] = []
    for source_id, onset in enumerate(sorted(onsets, key=lambda event: event.frame)):
        azimuth = _as_int_label(onset.azimuth, "azimuth")
        elevation = _as_int_label(onset.elevation, "elevation")
        distance = _as_int_label(onset.distance, "distance")
        for offset in range(active_frames):
            rows.append(
                MetadataRow(
                    frame=onset.frame + offset,
                    class_id=onset.class_id,
                    source_id=source_id,
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=distance,
                )
            )
    return tuple(rows)


def metadata_text_from_onsets(onsets: Iterable[OnsetEvent], *, active_frames: int = 1) -> str:
    """Serialize onset labels to native STARSS23 metadata text (dense per-frame activity rows)."""

    return format_starss23_metadata(metadata_rows_from_onsets(onsets, active_frames=active_frames))


# ---------------------------------------------------------------------------
# Audio identity and transport accounting.
# ---------------------------------------------------------------------------


def audio_sha256(audio: np.ndarray) -> str:
    """Content digest of a FOA audio array over canonical little-endian float32 bytes."""

    array = np.ascontiguousarray(np.asarray(audio), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_audio(audio: np.ndarray, clip_id: str) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 2 or array.shape[0] != N_CHANNELS:
        raise AdapterRefusal(f"clip {clip_id} audio must be shape ({N_CHANNELS}, n_samples)")
    if array.shape[1] == 0 or array.shape[1] % SAMPLES_PER_FRAME != 0:
        raise AdapterRefusal(f"clip {clip_id} audio must be a whole number of 100 ms frames")
    return array


# ---------------------------------------------------------------------------
# The native room-disjoint dev split.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NativeDevSplit:
    """The native STARSS23 room-disjoint dev-train / dev-test split (the 90 / 78 clip discipline)."""

    dev_train: tuple[str, ...]
    dev_test: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.dev_train) & set(self.dev_test)
        if overlap:
            raise AdapterRefusal(f"dev-train and dev-test share clips: {sorted(overlap)}")

    @property
    def sizes(self) -> dict[str, int]:
        return {"dev_train": len(self.dev_train), "dev_test": len(self.dev_test)}


# ---------------------------------------------------------------------------
# The adapter protocol and its two implementations.
# ---------------------------------------------------------------------------


@runtime_checkable
class StarssAdapter(Protocol):
    """The read seam every scored arm goes through, regardless of synthetic or real provenance."""

    def source_kind(self) -> str: ...

    def rights_clean(self) -> bool: ...

    def clips(self) -> tuple[Clip, ...]: ...

    def audio(self, clip_id: str) -> np.ndarray: ...

    def dev_split(self) -> NativeDevSplit: ...


def native_dev_split(clips: Sequence[Clip]) -> NativeDevSplit:
    """Project the one native fold-3 train / fold-4 test authority from clip identities."""

    dev_train: list[str] = []
    dev_test: list[str] = []
    train_rooms: set[str] = set()
    test_rooms: set[str] = set()
    for clip in clips:
        name = parse_clip_name(clip.clip_id)
        if name.fold == FOLD_DEV_TRAIN:
            dev_train.append(clip.clip_id)
            train_rooms.add(clip.room_id)
        elif name.fold == FOLD_DEV_TEST:
            dev_test.append(clip.clip_id)
            test_rooms.add(clip.room_id)
        else:
            raise AdapterRefusal(
                f"clip {clip.clip_id} is not in a STARSS23 dev fold ({FOLD_DEV_TRAIN} or {FOLD_DEV_TEST})"
            )
    shared_rooms = train_rooms & test_rooms
    if shared_rooms:
        raise AdapterRefusal(f"dev split is not room-disjoint, shared rooms: {sorted(shared_rooms)}")
    return NativeDevSplit(dev_train=tuple(sorted(dev_train)), dev_test=tuple(sorted(dev_test)))


def native_fold_split(
    adapter: StarssAdapter,
    n_val_rooms: int,
    *,
    refusal: type[Exception] = AdapterRefusal,
    refuse_empty: bool = True,
) -> ClipSplit:
    """Carve validation rooms from fold 3 while keeping fold 4 as the exact score partition."""

    dev = adapter.dev_split()
    by_id = {clip.clip_id: clip for clip in adapter.clips()}
    fold3 = [by_id[clip_id] for clip_id in dev.dev_train]
    fold4 = [by_id[clip_id] for clip_id in dev.dev_test]
    fold3_rooms = sorted({clip.room_id for clip in fold3})
    if n_val_rooms <= 0 or n_val_rooms >= len(fold3_rooms):
        raise refusal(
            f"n_val_rooms must leave at least one train room; saw {n_val_rooms} of {len(fold3_rooms)}"
        )
    val_rooms = set(fold3_rooms[-n_val_rooms:])
    train = tuple(sorted((clip for clip in fold3 if clip.room_id not in val_rooms), key=lambda x: x.clip_id))
    val = tuple(sorted((clip for clip in fold3 if clip.room_id in val_rooms), key=lambda x: x.clip_id))
    test = tuple(sorted(fold4, key=lambda x: x.clip_id))
    if refuse_empty and (not train or not val or not test):
        raise refusal("the fold-respecting split produced an empty partition")
    return ClipSplit(
        train=train,
        val=val,
        test=test,
        detail={
            "train_rooms": sorted({clip.room_id for clip in train}),
            "val_rooms": sorted(val_rooms),
            "test_rooms": sorted({clip.room_id for clip in test}),
            "split_rule": "test = native fold-4 dev-test; val = last N fold-3 rooms; train = rest of fold-3",
        },
    )


def map_clip_audio(
    adapter: StarssAdapter, transform: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    """Apply one frozen provider once to every clip's audio, keyed by stable clip identity."""

    return {clip.clip_id: transform(adapter.audio(clip.clip_id)) for clip in adapter.clips()}


def onset_density(clips: Sequence[Clip]) -> float:
    """Return label-only onset density over a clip sequence."""

    frames = sum(clip.n_frames for clip in clips)
    return sum(len(clip.onsets) for clip in clips) / frames if frames > 0 else 0.0


class ZeroParameterProvider:
    """Shared introspection for deterministic providers with no trained parameters."""

    __slots__ = ()

    def n_params(self) -> int:
        return 0


class FrozenFeatureProvider(ZeroParameterProvider):
    """Shared feature bytes and analytic per-frame cost introspection."""

    __slots__ = ()
    _flops_per_frame = 0
    _frame_count_refusal: type[ValueError] = ValueError

    def flops_for_frames(self, n_frames: int) -> int:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise self._frame_count_refusal("n_frames must be a nonnegative integer")
        return self._flops_per_frame * n_frames

    def feature_digest(self, features: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(features, dtype="<f8").tobytes()).hexdigest()


def domain_seed(seed: int, key: str, domain: bytes) -> int:
    """Return a uint32 seed separated by an explicit byte domain and semantic key."""

    payload = json.dumps(
        {"seed": int(seed), "key": str(key)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(domain + b"\0" + payload).digest()[:4], "big")


def marginal_matched_noise(
    noise_seed: int,
    n_frames: int,
    featurizer: Any,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Featurize deterministic STARSS-shaped white noise and match test feature marginals."""

    rng = np.random.default_rng(noise_seed)
    audio = rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))
    features = featurizer.featurize(audio)
    mean = float(features.mean())
    std = float(features.std())
    if std > 0.0:
        features = (features - mean) / std * float(target_std) + float(target_mean)
    return features


class SyntheticStarssAdapter:
    """Serve the frozen ``Clip`` contract from deterministic in-memory fixtures.

    Clips are built through the same parse path a real adapter uses: metadata text is parsed to rows,
    rows are reduced to onset labels, and the audio digest is recomputed from the media. ``source_kind``
    is ``"synthetic"`` so no run through this adapter can be promoted past a mechanics demonstration.
    """

    def __init__(
        self,
        audio_by_clip: Mapping[str, np.ndarray],
        metadata_by_clip: Mapping[str, str],
        *,
        rights_clean: bool = True,
    ) -> None:
        if set(audio_by_clip) != set(metadata_by_clip):
            raise AdapterRefusal("audio and metadata must cover exactly the same clip ids")
        if not audio_by_clip:
            raise AdapterRefusal("synthetic adapter needs at least one clip")
        self._rights_clean = bool(rights_clean)
        self._audio: dict[str, np.ndarray] = {}
        clips: list[Clip] = []
        for clip_id in sorted(audio_by_clip):
            name = parse_clip_name(clip_id)
            array = _require_audio(audio_by_clip[clip_id], clip_id)
            stored = np.ascontiguousarray(array, dtype="<f4")
            self._audio[clip_id] = stored
            onsets = onset_events_from_rows(parse_starss23_metadata(metadata_by_clip[clip_id]))
            clips.append(
                Clip(
                    clip_id=clip_id,
                    room_id=name.room_id,
                    n_frames=stored.shape[1] // SAMPLES_PER_FRAME,
                    audio_sha256=audio_sha256(stored),
                    onsets=onsets,
                )
            )
        self._clips: tuple[Clip, ...] = tuple(clips)
        self._by_id: dict[str, Clip] = {clip.clip_id: clip for clip in self._clips}

    def source_kind(self) -> str:
        return SOURCE_KIND_SYNTHETIC

    def rights_clean(self) -> bool:
        return self._rights_clean

    def clips(self) -> tuple[Clip, ...]:
        return self._clips

    def clip(self, clip_id: str) -> Clip:
        if clip_id not in self._by_id:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._by_id[clip_id]

    def onsets(self, clip_id: str) -> tuple[OnsetEvent, ...]:
        return self.clip(clip_id).onsets

    def audio(self, clip_id: str) -> np.ndarray:
        if clip_id not in self._audio:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._audio[clip_id]

    def dev_split(self) -> NativeDevSplit:
        return native_dev_split(self._clips)

    def harness_split(self, *, n_train_rooms: int, n_val_rooms: int) -> ClipSplit:
        """Build the three-way fit / tune / score partition through the schema's room-disjoint split.

        With ``n_train_rooms + n_val_rooms`` equal to the fold-3 room count, the score partition is
        exactly the fold-4 dev-test rooms, so the harness split nests inside the native dev split.
        """

        return room_disjoint_split(
            self._clips, n_train_rooms=n_train_rooms, n_val_rooms=n_val_rooms
        )

    def transport_charge(self) -> WorkVector:
        """Raw transport work for recomputing every clip's ``audio_sha256``, off the arm budget."""

        total_bytes = sum(array.nbytes for array in self._audio.values())
        return WorkVector(raw_transport_and_adapters=int(total_bytes))

    @classmethod
    def from_dir(cls, root: str | Path, *, rights_clean: bool = True) -> SyntheticStarssAdapter:
        """Load a STARSS23-shaped tree: ``foa_dev/<clip>.npy`` audio and ``metadata_dev/<clip>.csv``.

        This is the on-disk analogue of the real slot-in path: a real lane substitutes ``.wav`` decode
        for ``np.load`` and reaches the identical metadata parse and Clip construction.
        """

        root = Path(root)
        audio_dir = root / "foa_dev"
        meta_dir = root / "metadata_dev"
        if not meta_dir.is_dir() or not audio_dir.is_dir():
            raise AdapterRefusal(f"{root} is not a STARSS23-shaped tree (need foa_dev/ and metadata_dev/)")
        audio_by_clip: dict[str, np.ndarray] = {}
        metadata_by_clip: dict[str, str] = {}
        for meta_path in sorted(meta_dir.glob("*.csv")):
            clip_id = meta_path.stem
            audio_path = audio_dir / f"{clip_id}.npy"
            if not audio_path.exists():
                raise AdapterRefusal(f"metadata for {clip_id} has no matching audio at {audio_path}")
            audio_by_clip[clip_id] = np.load(audio_path, allow_pickle=False)
            metadata_by_clip[clip_id] = meta_path.read_text(encoding="utf-8")
        if not metadata_by_clip:
            raise AdapterRefusal(f"no clips found under {root}")
        return cls(audio_by_clip, metadata_by_clip, rights_clean=rights_clean)


# STARSS23 FOA media is delivered as 24 kHz, 4-channel, 16-bit PCM WAV. These are corpus constants that
# the real decode asserts, not knobs: a file that violates them is not the STARSS23 FOA contract.
_REAL_SAMPLE_RATE_HZ = SAMPLE_RATE_HZ
_REAL_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
_INT16_FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class ClipTruncation:
    """Provenance of the whole-frame truncation applied to one real clip on decode."""

    clip_id: str
    raw_samples: int
    kept_frames: int
    dropped_tail_samples: int
    dropped_onsets_past_end: int
    capped_by_max_frames: bool

    def payload(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "raw_samples": self.raw_samples,
            "kept_frames": self.kept_frames,
            "dropped_tail_samples": self.dropped_tail_samples,
            "dropped_onsets_past_end": self.dropped_onsets_past_end,
            "capped_by_max_frames": self.capped_by_max_frames,
        }


def decode_foa_wav(path: str | Path) -> np.ndarray:
    """Decode a STARSS23 FOA WAV into a ``(N_CHANNELS, n_samples)`` float array in [-1, 1).

    The file must be 24 kHz, four channel, 16-bit PCM (the frozen STARSS23 FOA acquisition grid). Samples
    are read with the stdlib ``wave`` module, de-interleaved to channel-major order, and scaled from
    int16 to float by the int16 full-scale so the featurizer sees a consistent amplitude across clips.
    No resampling and no channel remap is performed: the corpus is already on the bed's native grid.
    """

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
        n_audio_frames = handle.getnframes()
        raw = handle.readframes(n_audio_frames)
    if n_channels != N_CHANNELS:
        raise AdapterRefusal(f"{path} has {n_channels} channels, STARSS23 FOA needs {N_CHANNELS}")
    if sample_width != _REAL_SAMPLE_WIDTH_BYTES:
        raise AdapterRefusal(f"{path} is not 16-bit PCM (sample width {sample_width} bytes)")
    if frame_rate != _REAL_SAMPLE_RATE_HZ:
        raise AdapterRefusal(f"{path} is {frame_rate} Hz, STARSS23 FOA needs {_REAL_SAMPLE_RATE_HZ} Hz")
    interleaved = np.frombuffer(raw, dtype="<i2")
    if interleaved.size % N_CHANNELS != 0:
        raise AdapterRefusal(f"{path} sample count is not a whole number of {N_CHANNELS}-channel frames")
    channel_major = interleaved.reshape(-1, N_CHANNELS).T
    return channel_major.astype(np.float64) / _INT16_FULL_SCALE


def _truncate_to_frames(audio: np.ndarray, max_frames: int | None) -> tuple[np.ndarray, int, bool]:
    """Truncate a decoded clip to a whole number of 100 ms frames (and an optional frame cap)."""

    n_samples = audio.shape[1]
    n_frames = n_samples // SAMPLES_PER_FRAME
    capped = False
    if max_frames is not None and n_frames > max_frames:
        n_frames = max_frames
        capped = True
    if n_frames <= 0:
        raise AdapterRefusal("a real clip must carry at least one whole 100 ms frame")
    kept_samples = n_frames * SAMPLES_PER_FRAME
    return np.ascontiguousarray(audio[:, :kept_samples], dtype="<f4"), n_frames, capped


class RealStarssAdapter:
    """Serve the frozen ``Clip`` contract from a real, MIT-licensed STARSS23 FOA tree.

    This is the real-data twin of ``SyntheticStarssAdapter``. It decodes the native 24 kHz, 4-channel,
    16-bit FOA WAV media and parses the native STARSS23 metadata CSVs through the exact same parse and
    onset-derivation path the synthetic adapter uses, so the harness and referee consume it unchanged.
    ``source_kind`` is ``"real"`` and ``rights_clean`` defaults to True for the rights-clean MIT corpus.

    Construction points at two directory roots (the FOA audio tree and the metadata tree). Every ``*.wav``
    reachable under ``foa_root`` is matched to its ``<clip>.csv`` under ``metadata_root`` by clip stem, so
    the STARSS23 subset layout ``foa_dev/dev-train-*/`` and ``metadata_dev/dev-train-*/`` slots in with no
    reshaping. Each clip is truncated to a whole number of 100 ms label frames (and an optional
    ``max_frames`` cap for the FLOP budget); metadata onsets past the kept length are dropped and the
    truncation is recorded per clip so nothing is silently lost.
    """

    def __init__(
        self,
        foa_root: str | Path,
        metadata_root: str | Path,
        *,
        rights_clean: bool = True,
        max_frames: int | None = None,
    ) -> None:
        self._foa_root = Path(foa_root)
        self._metadata_root = Path(metadata_root)
        self._rights_clean = bool(rights_clean)
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int)):
            raise AdapterRefusal("max_frames must be an integer or None")
        if max_frames is not None and max_frames <= 0:
            raise AdapterRefusal("max_frames must be a positive integer when given")
        if not self._foa_root.is_dir():
            raise AdapterRefusal(f"FOA audio root {self._foa_root} is not a directory")
        if not self._metadata_root.is_dir():
            raise AdapterRefusal(f"metadata root {self._metadata_root} is not a directory")

        meta_by_stem: dict[str, Path] = {}
        for meta_path in sorted(self._metadata_root.rglob("*.csv")):
            meta_by_stem.setdefault(meta_path.stem, meta_path)

        wav_paths = sorted(self._foa_root.rglob("*.wav"))
        if not wav_paths:
            raise AdapterRefusal(f"no FOA WAV clips found under {self._foa_root}")

        self._audio: dict[str, np.ndarray] = {}
        self._truncations: list[ClipTruncation] = []
        clips: list[Clip] = []
        for wav_path in wav_paths:
            clip_id = wav_path.stem
            name = parse_clip_name(clip_id)  # refuses any non STARSS23 filename before any decode
            meta_path = meta_by_stem.get(clip_id)
            if meta_path is None:
                raise AdapterRefusal(
                    f"FOA clip {clip_id} has no matching metadata under {self._metadata_root}"
                )
            decoded = decode_foa_wav(wav_path)
            raw_samples = decoded.shape[1]
            audio, n_frames, capped = _truncate_to_frames(decoded, max_frames)
            self._audio[clip_id] = audio

            metadata_text = meta_path.read_text(encoding="utf-8")
            all_onsets = onset_events_from_rows(parse_starss23_metadata(metadata_text))
            kept_onsets = tuple(onset for onset in all_onsets if onset.frame < n_frames)
            self._truncations.append(
                ClipTruncation(
                    clip_id=clip_id,
                    raw_samples=raw_samples,
                    kept_frames=n_frames,
                    dropped_tail_samples=raw_samples - n_frames * SAMPLES_PER_FRAME,
                    dropped_onsets_past_end=len(all_onsets) - len(kept_onsets),
                    capped_by_max_frames=capped,
                )
            )
            clips.append(
                Clip(
                    clip_id=clip_id,
                    room_id=name.room_id,
                    n_frames=n_frames,
                    audio_sha256=audio_sha256(audio),
                    onsets=kept_onsets,
                )
            )
        self._clips: tuple[Clip, ...] = tuple(clips)
        self._by_id: dict[str, Clip] = {clip.clip_id: clip for clip in self._clips}

    def source_kind(self) -> str:
        return SOURCE_KIND_REAL

    def rights_clean(self) -> bool:
        return self._rights_clean

    def clips(self) -> tuple[Clip, ...]:
        return self._clips

    def clip(self, clip_id: str) -> Clip:
        if clip_id not in self._by_id:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._by_id[clip_id]

    def onsets(self, clip_id: str) -> tuple[OnsetEvent, ...]:
        return self.clip(clip_id).onsets

    def audio(self, clip_id: str) -> np.ndarray:
        if clip_id not in self._audio:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._audio[clip_id]

    def truncations(self) -> tuple[ClipTruncation, ...]:
        """Per-clip whole-frame truncation provenance, so no clip is silently shortened or dropped."""

        return tuple(self._truncations)

    def dev_split(self) -> NativeDevSplit:
        return native_dev_split(self._clips)

    def transport_charge(self) -> WorkVector:
        """Raw transport work for recomputing every clip's ``audio_sha256``, off the arm budget."""

        total_bytes = sum(array.nbytes for array in self._audio.values())
        return WorkVector(raw_transport_and_adapters=int(total_bytes))
