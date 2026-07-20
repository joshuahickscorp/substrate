from __future__ import annotations

import hashlib
import json
import re
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    Clip,
    ClipSplit,
)

FOLD_DEV_TRAIN = 3
FOLD_DEV_TEST = 4

_AZIMUTH_MIN, _AZIMUTH_MAX = -180, 180
_ELEVATION_MIN, _ELEVATION_MAX = -90, 90
_DISTANCE_MAX_CM = 100_000  # 1 km ceiling; a physical sanity bound, not a knob
_COUNT_CEILING = 16

_CLIP_NAME_RE = re.compile(r"^fold(\d+)_room(\d+)_mix(\d+)$")


class AdapterRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClipName:
    fold: int
    room: int
    mix: int

    def __post_init__(self) -> None:
        for name, value in (("fold", self.fold), ("room", self.room), ("mix", self.mix)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AdapterRefusal(f"ClipName.{name} must be a nonnegative integer")

    @property
    def room_id(self) -> str:
        return f"room{self.room:02d}"


def parse_clip_name(name: str) -> ClipName:

    if not isinstance(name, str) or not name.strip():
        raise AdapterRefusal("clip name must be a nonempty string")
    stem = Path(name.strip()).stem if "." in name else name.strip()
    match = _CLIP_NAME_RE.fullmatch(stem)
    if match is None:
        raise AdapterRefusal(f"clip name {name!r} is not fold<F>_room<R>_mix<M>")
    return ClipName(fold=int(match.group(1)), room=int(match.group(2)), mix=int(match.group(3)))


@dataclass(frozen=True, slots=True)
class MetadataRow:
    frame: int
    class_id: int
    source_id: int
    has_distance: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("frame", self.frame),
            ("class_id", self.class_id),
            ("source_id", self.source_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdapterRefusal(f"MetadataRow.{name} must be an integer")
        if self.frame < 0:
            raise AdapterRefusal("MetadataRow.frame must be nonnegative")
        if not 0 <= self.class_id < N_CLASSES:
            raise AdapterRefusal(f"MetadataRow.class_id must be in [0, {N_CLASSES})")
        if self.source_id < 0:
            raise AdapterRefusal("MetadataRow.source_id must be nonnegative")


def parse_starss23_metadata(text: str) -> tuple[MetadataRow, ...]:

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
        frame, class_id, source_id, azimuth, elevation, *distance = values
        row = MetadataRow(frame, class_id, source_id, bool(distance))
        if not _AZIMUTH_MIN <= azimuth <= _AZIMUTH_MAX:
            raise AdapterRefusal("MetadataRow.azimuth must lie in [-180, 180] degrees")
        if not _ELEVATION_MIN <= elevation <= _ELEVATION_MAX:
            raise AdapterRefusal("MetadataRow.elevation must lie in [-90, 90] degrees")
        if distance and not 0 < distance[0] <= _DISTANCE_MAX_CM:
            raise AdapterRefusal("MetadataRow.distance must be a positive centimeter count")
        rows.append(row)
    return tuple(rows)


def _count_track(rows: Iterable[MetadataRow], n_frames: int) -> tuple[int, ...]:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise AdapterRefusal("n_frames must be a positive integer")
    active: list[set[tuple[int, int]]] = [set() for _ in range(n_frames)]
    for row in rows:
        if not isinstance(row, MetadataRow):
            raise AdapterRefusal("rows must be MetadataRow values")
        if row.frame < n_frames:
            active[row.frame].add((row.class_id, row.source_id))
    track = tuple(len(sources) for sources in active)
    if track and max(track) > _COUNT_CEILING:
        raise AdapterRefusal(f"derived concurrent count {max(track)} exceeds COUNT_CEILING {_COUNT_CEILING}")
    return track


def _onset_frames(rows: Iterable[MetadataRow]) -> set[int]:
    by_track: dict[tuple[int, int], list[MetadataRow]] = {}
    for row in rows:
        if not row.has_distance:
            raise AdapterRefusal("onset derivation requires STARSS23 distance labels")
        by_track.setdefault((row.class_id, row.source_id), []).append(row)
    frames: set[int] = set()
    for track_rows in by_track.values():
        track_rows.sort(key=lambda row: row.frame)
        previous: int | None = None
        for row in track_rows:
            if previous is None or row.frame != previous + 1:
                frames.add(row.frame)
            previous = row.frame
    return frames


def audio_sha256(audio: np.ndarray) -> str:

    array = np.ascontiguousarray(np.asarray(audio), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def native_fold_split(
    adapter: RealStarssAdapter,
    n_val_rooms: int,
    *,
    refusal: type[Exception] = AdapterRefusal,
    refuse_empty: bool = True,
) -> ClipSplit:

    fold3: list[Clip] = []
    fold4: list[Clip] = []
    for clip in adapter.clips():
        fold = parse_clip_name(clip.clip_id).fold
        if fold == FOLD_DEV_TRAIN:
            fold3.append(clip)
        elif fold == FOLD_DEV_TEST:
            fold4.append(clip)
        else:
            raise AdapterRefusal(
                f"clip {clip.clip_id} is not in a STARSS23 dev fold ({FOLD_DEV_TRAIN} or {FOLD_DEV_TEST})"
            )
    shared_rooms = {clip.room_id for clip in fold3} & {clip.room_id for clip in fold4}
    if shared_rooms:
        raise AdapterRefusal(f"dev split is not room-disjoint, shared rooms: {sorted(shared_rooms)}")
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
    adapter: RealStarssAdapter, transform: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:

    return {clip.clip_id: transform(adapter.audio(clip.clip_id)) for clip in adapter.clips()}


class ZeroParameterProvider:
    __slots__ = ()

    def n_params(self) -> int:
        return 0


def domain_seed(seed: int, key: str, domain: bytes) -> int:

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

    rng = np.random.default_rng(noise_seed)
    audio = rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))
    features = featurizer.featurize(audio)
    mean = float(features.mean())
    std = float(features.std())
    if std > 0.0:
        features = (features - mean) / std * float(target_std) + float(target_mean)
    return features


_REAL_SAMPLE_RATE_HZ = SAMPLE_RATE_HZ
_REAL_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
_INT16_FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class ClipTruncation:
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
    def __init__(
        self,
        foa_root: str | Path,
        metadata_root: str | Path,
        *,
        max_frames: int | None = None,
    ) -> None:
        self._foa_root = Path(foa_root)
        self._metadata_root = Path(metadata_root)
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
        self._count_tracks: dict[str, tuple[int, ...]] = {}
        self._truncations: list[ClipTruncation] = []
        clips: list[Clip] = []
        for wav_path in wav_paths:
            clip_id = wav_path.stem
            name = parse_clip_name(clip_id)  # refuses any non STARSS23 filename before any decode
            matching_meta_path = meta_by_stem.get(clip_id)
            if matching_meta_path is None:
                raise AdapterRefusal(
                    f"FOA clip {clip_id} has no matching metadata under {self._metadata_root}"
                )
            decoded = decode_foa_wav(wav_path)
            raw_samples = decoded.shape[1]
            audio, n_frames, capped = _truncate_to_frames(decoded, max_frames)
            self._audio[clip_id] = audio

            metadata_rows = parse_starss23_metadata(matching_meta_path.read_text(encoding="utf-8"))
            onset_frames = _onset_frames(metadata_rows)
            self._count_tracks[clip_id] = _count_track(metadata_rows, n_frames)
            self._truncations.append(
                ClipTruncation(
                    clip_id=clip_id,
                    raw_samples=raw_samples,
                    kept_frames=n_frames,
                    dropped_tail_samples=raw_samples - n_frames * SAMPLES_PER_FRAME,
                    dropped_onsets_past_end=sum(frame >= n_frames for frame in onset_frames),
                    capped_by_max_frames=capped,
                )
            )
            clips.append(
                Clip(
                    clip_id=clip_id,
                    room_id=name.room_id,
                    n_frames=n_frames,
                    audio_sha256=audio_sha256(audio),
                )
            )
        self._clips: tuple[Clip, ...] = tuple(clips)

    def clips(self) -> tuple[Clip, ...]:
        return self._clips

    def audio(self, clip_id: str) -> np.ndarray:
        if clip_id not in self._audio:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._audio[clip_id]

    def count_track(self, clip_id: str) -> tuple[int, ...]:
        if clip_id not in self._count_tracks:
            raise AdapterRefusal(f"unknown clip id {clip_id!r}")
        return self._count_tracks[clip_id]

    def truncations(self) -> tuple[ClipTruncation, ...]:

        return tuple(self._truncations)
