"""Component 1c: the STARSS23-metadata-faithful synthetic corpus generator.

This is the fixture generator the data adapter is scored through. It complements the regime-oriented
per-clip primitives in ``fixtures.py`` by producing exactly what a real STARSS23 download provides and
what ``adapter.py`` parses: 4-channel FOA audio at 24 kHz plus native integer metadata label files, laid
out room-disjoint across fold-3 dev-train and fold-4 dev-test rooms. Labels use the real STARSS23 units
(integer degrees of azimuth and elevation, integer centimeters of distance) so they round-trip through
the metadata schema exactly; the regime fixtures use continuous floats and never touch the label files.

Every array is byte-reproducible from a single integer seed. All randomness comes from PCG64 streams
domain-separated per room and per clip through ``mop.seeding.derive_seed32``. Three faithfulness
properties are built into the signal, matching the deep-research recipe:

- Planted onsets are coherent FOA point sources steered by the exact analytic first-order encoding
  (W = 1/sqrt(2), X = cos(az) cos(el), Y = sin(az) cos(el), Z = sin(el)) with an inverse-distance law,
  so firing at the planted frames scores onset F1 = 1.0.
- The background is room-correlated: each room owns a fixed FIR coloration, so a model that memorizes
  one room's texture does not transfer to a disjoint room. That makes the room split a real test.
- Nuisance flux is spatially incoherent energy at non-onset frames. It raises single-channel flux
  exactly where there is no onset, so a bare flux threshold cannot reach F1 = 1.0; only a
  coherence-aware gate separates onsets from nuisance. That keeps the task off the trivial ceiling.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mop.seeding import derive_seed32

from .adapter import (
    MetadataRow,
    SyntheticStarssAdapter,
    format_clip_id,
    format_starss23_metadata,
)
from .schema import (
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    ClipSplit,
    OnsetEvent,
    room_disjoint_split,
)

# FOA encoding constants. W carries the FuMa 1/sqrt(2) weight; the radial law anchors the spec's
# inverse-distance falloff at a 1 m reference so centimeter distances give sane amplitudes.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_REF_DISTANCE_CM = 100.0

# Grain and burst shaping. The sharp attack at sample zero is what makes an onset a flux spike.
_DECAY_TAU_S = 0.02
_TONE_NOISE_MIX = 0.3
_ROOM_KERNEL_LEN = 12
# One class-characteristic tone per STARSS23 class, log-spaced across the speech-to-transient band.
_CLASS_BASE_HZ = tuple(float(hz) for hz in np.geomspace(200.0, 6000.0, N_CLASSES))

# Provenance floors from Q3(d): at least 12 test and 24 validation positives for a stable rate estimate.
MIN_TEST_ONSETS = 12
MIN_VAL_ONSETS = 24


class CorpusRefusal(ValueError):
    """Raised when a synthetic corpus configuration is internally inconsistent."""


def _derive_child(master_seed: int, *parts: object) -> int:
    """Domain-separated 32-bit child seed for one purpose, stable across hosts.

    ``derive_seed32`` returns an in-range seed unchanged, so a small master seed is first lifted above
    2**32; that forces the hashed, namespace-mixing path, yielding a child that depends on both the
    master seed and the namespace. PCG64 is then byte-identical across platforms for that child.
    """

    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise CorpusRefusal("master seed must be a nonnegative integer")
    namespace = "/".join(str(part) for part in parts)
    lifted = master_seed + (1 << 40)
    return derive_seed32(lifted, namespace)


@dataclass(frozen=True, slots=True)
class SyntheticCorpusConfig:
    """Knobs for the synthetic corpus. Defaults clear the Q3(d) validation and test positive floors.

    Rooms mirror the native STARSS23 dev discipline: ``n_fold3_rooms`` room-disjoint dev-train rooms
    (split further into fit and tune rooms) and ``n_fold4_rooms`` disjoint dev-test rooms. Real STARSS23
    is 90 dev-train and 78 dev-test clips; ``real_scale`` mirrors that shape and the 60 s clip length.
    """

    n_fold3_rooms: int = 5
    n_val_rooms: int = 2
    n_fold4_rooms: int = 2
    clips_per_room: int = 3
    clip_frames: int = 40
    onsets_per_clip: int = 6
    nuisances_per_clip: int = 6
    active_frames: int = 2
    distance_min_cm: int = 30
    distance_max_cm: int = 500
    onset_gain: float = 1.0
    nuisance_gain: float = 1.0
    background_gain: float = 0.08

    def __post_init__(self) -> None:
        positive_ints = {
            "n_fold3_rooms": self.n_fold3_rooms,
            "n_val_rooms": self.n_val_rooms,
            "n_fold4_rooms": self.n_fold4_rooms,
            "clips_per_room": self.clips_per_room,
            "clip_frames": self.clip_frames,
            "onsets_per_clip": self.onsets_per_clip,
            "nuisances_per_clip": self.nuisances_per_clip,
            "active_frames": self.active_frames,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CorpusRefusal(f"SyntheticCorpusConfig.{name} must be a positive integer")
        if self.n_val_rooms >= self.n_fold3_rooms:
            raise CorpusRefusal("n_val_rooms must leave at least one fold-3 room for training")
        if self.active_frames >= self.clip_frames:
            raise CorpusRefusal("active_frames must be shorter than clip_frames")
        if not 0 < self.distance_min_cm <= self.distance_max_cm:
            raise CorpusRefusal("distance range must satisfy 0 < min <= max")
        for gain_name, gain_value in (
            ("onset_gain", self.onset_gain),
            ("nuisance_gain", self.nuisance_gain),
            ("background_gain", self.background_gain),
        ):
            if (
                not isinstance(gain_value, (int, float))
                or isinstance(gain_value, bool)
                or not gain_value > 0
            ):
                raise CorpusRefusal(f"SyntheticCorpusConfig.{gain_name} must be a positive number")
        events = self.onsets_per_clip + self.nuisances_per_clip
        valid_starts = self.clip_frames - self.active_frames + 1
        if valid_starts // events < self.active_frames:
            raise CorpusRefusal(
                "clip_frames is too short to place non-overlapping events: need "
                f"{events * self.active_frames} valid start frames, have {valid_starts}"
            )

    @property
    def n_train_rooms(self) -> int:
        return self.n_fold3_rooms - self.n_val_rooms

    @classmethod
    def tiny(cls) -> SyntheticCorpusConfig:
        """A minimal fast corpus for unit tests. Positive floors are not guaranteed at this size."""

        return cls(
            n_fold3_rooms=3,
            n_val_rooms=1,
            n_fold4_rooms=1,
            clips_per_room=1,
            clip_frames=10,
            onsets_per_clip=2,
            nuisances_per_clip=2,
            active_frames=2,
        )

    @classmethod
    def real_scale(cls) -> SyntheticCorpusConfig:
        """Mirror the STARSS23 dev shape and 60 s clip length for the matched-budget harness lane."""

        return cls(
            n_fold3_rooms=9,
            n_val_rooms=2,
            n_fold4_rooms=6,
            clips_per_room=10,
            clip_frames=600,
            onsets_per_clip=40,
            nuisances_per_clip=40,
            active_frames=2,
        )


def _room_kernel(room_rng: np.random.Generator) -> np.ndarray:
    """A fixed per-room FIR coloration. Same room and master seed always yield the same kernel."""

    kernel = room_rng.standard_normal(_ROOM_KERNEL_LEN)
    norm = float(np.linalg.norm(kernel))
    if norm == 0.0:
        kernel[0] = 1.0
        norm = 1.0
    return kernel / norm


def _onset_grain(rng: np.random.Generator, length: int, class_id: int) -> np.ndarray:
    """A mono onset transient: sharp attack, exponential decay, class-characteristic tone plus noise."""

    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE_HZ
    envelope = np.exp(-t / _DECAY_TAU_S)
    tone = np.sin(2.0 * np.pi * _CLASS_BASE_HZ[class_id] * t)
    noise = rng.standard_normal(length)
    return envelope * (tone + _TONE_NOISE_MIX * noise)


def _foa_encode(grain: np.ndarray, azimuth: int, elevation: int, distance_cm: int) -> np.ndarray:
    """Steer a mono grain into a coherent FOA point source, shape (N_CHANNELS, length).

    Exact spec encoding: W = 1/sqrt(2), X = cos(az) cos(el), Y = sin(az) cos(el), Z = sin(el), each
    scaled by an inverse-distance radial gain anchored at the 1 m reference. The four channels are
    scaled copies of one grain, so the source is perfectly inter-channel coherent.
    """

    az = math.radians(azimuth)
    el = math.radians(elevation)
    radial = _REF_DISTANCE_CM / float(max(distance_cm, 1))
    coefficients = np.array(
        [
            _INV_SQRT2,
            math.cos(az) * math.cos(el),
            math.sin(az) * math.cos(el),
            math.sin(el),
        ],
        dtype=np.float64,
    )
    return coefficients[:, None] * (radial * grain)[None, :]


def _nuisance_burst(rng: np.random.Generator, length: int) -> np.ndarray:
    """A spatially incoherent energy burst, shape (N_CHANNELS, length).

    Same decaying envelope as an onset grain, so it raises the spectral flux, but each channel is
    independent noise, so it is not a valid FOA point source and a coherence-aware gate can reject it.
    """

    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE_HZ
    envelope = np.exp(-t / _DECAY_TAU_S)
    channels = [envelope * rng.standard_normal(length) for _ in range(N_CHANNELS)]
    return np.stack(channels, axis=0)


def _spaced_event_frames(
    rng: np.random.Generator, count: int, clip_frames: int, active_frames: int
) -> list[int]:
    """Pick ``count`` non-overlapping event start frames, one jittered frame per equal bin."""

    valid_starts = clip_frames - active_frames + 1
    bin_width = valid_starts // count
    frames: list[int] = []
    for index in range(count):
        low = index * bin_width
        span = bin_width - active_frames + 1
        offset = int(rng.integers(0, span)) if span > 1 else 0
        frames.append(low + offset)
    return frames


@dataclass(frozen=True, slots=True)
class SyntheticStarssCorpus:
    """Raw synthetic media and native label text plus the planted ground truth for verification."""

    config: SyntheticCorpusConfig
    seed: int
    clip_ids: tuple[str, ...]
    audio_by_clip: Mapping[str, np.ndarray]
    metadata_by_clip: Mapping[str, str]
    planted_by_clip: Mapping[str, tuple[OnsetEvent, ...]]
    nuisance_frames_by_clip: Mapping[str, tuple[int, ...]]
    detail: dict[str, object] = field(default_factory=dict)

    def audio(self, clip_id: str) -> np.ndarray:
        return self.audio_by_clip[clip_id]

    def metadata_text(self, clip_id: str) -> str:
        return self.metadata_by_clip[clip_id]

    def adapter(self) -> SyntheticStarssAdapter:
        """The parse seam over this corpus. Real STARSS23 uses the same adapter, unchanged."""

        return SyntheticStarssAdapter(self.audio_by_clip, self.metadata_by_clip)

    def default_split(self) -> ClipSplit:
        """The three-way room-disjoint and clip-disjoint fit / tune / score partition.

        Train and val come from the fold-3 dev-train rooms; test is the fold-4 dev-test rooms, so the
        score partition nests exactly inside the native STARSS23 dev split.
        """

        clips = self.adapter().clips()
        return room_disjoint_split(
            clips,
            n_train_rooms=self.config.n_train_rooms,
            n_val_rooms=self.config.n_val_rooms,
        )

    @staticmethod
    def positive_counts(split: ClipSplit) -> dict[str, int]:
        """Onset (positive) counts per partition, for checking the Q3(d) validation and test floors."""

        return {
            "train": sum(len(clip.onsets) for clip in split.train),
            "val": sum(len(clip.onsets) for clip in split.val),
            "test": sum(len(clip.onsets) for clip in split.test),
        }

    def write_to(self, root: str | Path) -> Path:
        """Write a STARSS23-shaped layout: ``foa_dev/<clip>.npy`` and ``metadata_dev/<clip>.csv``.

        Audio is stored as exact little-endian float32 ``.npy``; the real lane substitutes ``.wav`` and
        the same metadata files. This makes the on-disk round trip testable without an audio codec.
        """

        root = Path(root)
        audio_dir = root / "foa_dev"
        meta_dir = root / "metadata_dev"
        audio_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        for clip_id in self.clip_ids:
            array = np.ascontiguousarray(self.audio_by_clip[clip_id], dtype="<f4")
            np.save(audio_dir / f"{clip_id}.npy", array, allow_pickle=False)
            (meta_dir / f"{clip_id}.csv").write_text(self.metadata_by_clip[clip_id], encoding="utf-8")
        return root


def _build_clip_audio(
    config: SyntheticCorpusConfig,
    seed: int,
    clip_id: str,
    room_num: int,
) -> tuple[np.ndarray, str, tuple[OnsetEvent, ...], tuple[int, ...]]:
    """Synthesize one clip: audio, native metadata text, planted onsets, and nuisance frames."""

    clip_rng = np.random.default_rng(_derive_child(seed, "clip", clip_id))
    room_rng = np.random.default_rng(_derive_child(seed, "room", room_num))
    n_samples = config.clip_frames * SAMPLES_PER_FRAME
    grain_length = config.active_frames * SAMPLES_PER_FRAME
    audio = np.zeros((N_CHANNELS, n_samples), dtype=np.float64)

    # 1. Room-correlated background: one fixed room kernel colors independent per-channel noise.
    kernel = _room_kernel(room_rng)
    for channel in range(N_CHANNELS):
        white = clip_rng.standard_normal(n_samples)
        audio[channel] += config.background_gain * np.convolve(white, kernel, mode="same")

    # 2. Non-overlapping event start frames, then split into onset and nuisance frames.
    total_events = config.onsets_per_clip + config.nuisances_per_clip
    starts = _spaced_event_frames(clip_rng, total_events, config.clip_frames, config.active_frames)
    order = clip_rng.permutation(total_events)
    onset_frames = sorted(int(starts[i]) for i in order[: config.onsets_per_clip])
    nuisance_frames = sorted(int(starts[i]) for i in order[config.onsets_per_clip :])

    # 3. Plant coherent FOA onset point sources at the onset frames (integer degrees and centimeters).
    planted: list[OnsetEvent] = []
    for frame in onset_frames:
        class_id = int(clip_rng.integers(0, N_CLASSES))
        azimuth = int(clip_rng.integers(-180, 181))
        elevation = int(clip_rng.integers(-90, 91))
        distance = int(clip_rng.integers(config.distance_min_cm, config.distance_max_cm + 1))
        grain = _onset_grain(clip_rng, grain_length, class_id)
        foa = _foa_encode(grain, azimuth, elevation, distance)
        start = frame * SAMPLES_PER_FRAME
        audio[:, start : start + grain_length] += config.onset_gain * foa
        planted.append(
            OnsetEvent(
                frame=frame,
                class_id=class_id,
                azimuth=float(azimuth),
                elevation=float(elevation),
                distance=float(distance),
            )
        )

    # 4. Inject spatially incoherent nuisance flux at the non-onset frames.
    for frame in nuisance_frames:
        burst = _nuisance_burst(clip_rng, grain_length)
        start = frame * SAMPLES_PER_FRAME
        audio[:, start : start + grain_length] += config.nuisance_gain * burst

    # 5. Deterministic peak normalization keeps the array in range without changing relative structure.
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio * (0.9 / peak)
    audio32 = np.ascontiguousarray(audio, dtype=np.float32)

    # 6. Native STARSS23 dense metadata: one row per active frame per source, distance in centimeters.
    rows: list[MetadataRow] = []
    for source_id, event in enumerate(sorted(planted, key=lambda ev: ev.frame)):
        for offset in range(config.active_frames):
            rows.append(
                MetadataRow(
                    frame=event.frame + offset,
                    class_id=event.class_id,
                    source_id=source_id,
                    azimuth=int(event.azimuth),
                    elevation=int(event.elevation),
                    distance=int(event.distance),
                )
            )
    metadata_text = format_starss23_metadata(rows)
    planted_sorted = tuple(sorted(planted, key=lambda ev: ev.frame))
    return audio32, metadata_text, planted_sorted, tuple(sorted(nuisance_frames))


def generate_corpus(
    config: SyntheticCorpusConfig | None = None, *, seed: int = 0
) -> SyntheticStarssCorpus:
    """Deterministically synthesize the whole corpus. Same config and seed give byte-identical media."""

    config = config or SyntheticCorpusConfig()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CorpusRefusal("seed must be a nonnegative integer")

    # Fold-3 rooms take the low room numbers so they sort before the fold-4 dev-test rooms.
    fold3_rooms = list(range(config.n_fold3_rooms))
    fold4_rooms = list(range(config.n_fold3_rooms, config.n_fold3_rooms + config.n_fold4_rooms))
    room_folds = [(3, room) for room in fold3_rooms] + [(4, room) for room in fold4_rooms]

    clip_ids: list[str] = []
    audio_by_clip: dict[str, np.ndarray] = {}
    metadata_by_clip: dict[str, str] = {}
    planted_by_clip: dict[str, tuple[OnsetEvent, ...]] = {}
    nuisance_frames_by_clip: dict[str, tuple[int, ...]] = {}
    for fold, room_num in room_folds:
        for mix in range(config.clips_per_room):
            clip_id = format_clip_id(fold, room_num, mix)
            audio32, metadata_text, planted, nuisance_frames = _build_clip_audio(
                config, seed, clip_id, room_num
            )
            clip_ids.append(clip_id)
            audio_by_clip[clip_id] = audio32
            metadata_by_clip[clip_id] = metadata_text
            planted_by_clip[clip_id] = planted
            nuisance_frames_by_clip[clip_id] = nuisance_frames

    return SyntheticStarssCorpus(
        config=config,
        seed=seed,
        clip_ids=tuple(clip_ids),
        audio_by_clip=audio_by_clip,
        metadata_by_clip=metadata_by_clip,
        planted_by_clip=planted_by_clip,
        nuisance_frames_by_clip=nuisance_frames_by_clip,
        detail={
            "fold3_rooms": [f"room{r:02d}" for r in fold3_rooms],
            "fold4_rooms": [f"room{r:02d}" for r in fold4_rooms],
            "clips": len(clip_ids),
        },
    )


def frame_energy(audio: np.ndarray, *, channel: int = 0) -> np.ndarray:
    """Per-100 ms-frame energy on one channel. A minimal, featurizer-free onset proxy for tests."""

    array = np.asarray(audio, dtype=np.float64)
    n_frames = array.shape[1] // SAMPLES_PER_FRAME
    channel_signal = array[channel, : n_frames * SAMPLES_PER_FRAME]
    framed = channel_signal.reshape(n_frames, SAMPLES_PER_FRAME)
    return np.sum(framed * framed, axis=1)
