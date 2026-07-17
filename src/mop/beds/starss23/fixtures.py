"""Component 1c: the deterministic byte-reproducible synthetic STARSS23 generator.

Every array is generated from a domain-separated child seed, so the whole pipeline is testable now
without any real download. Planted onset grains are encoded with analytic first-order-Ambisonics
steering (W = 1/sqrt(2), X = cos(az)cos(el), Y = sin(az)cos(el), Z = sin(el), each divided by
distance), so a spatial oracle that knows the onset frames scores onset F1 = 1.0 on the favorable
regime. Rooms carry a correlated colored background so a room-disjoint split tests real
generalization, not room memorization. Nuisance grains are planted in a different spectral band with
matched energy, so a bare total-flux threshold trips on them (it cannot separate the bands) while a
gate that reads the per-mel-bin features can. That keeps the bed off the flux-threshold ceiling.

Compute here is charged to WorkVector.raw_transport_and_adapters, off the arm budget.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .schema import (
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    Clip,
    OnsetEvent,
)

# Two disjoint spectral bands with matched energy. Onsets live in the signal band, nuisance in the
# nuisance band. Total flux is comparable at both, so magnitude alone cannot separate them.
SIGNAL_BAND_HZ = (700.0, 1500.0)
NUISANCE_BAND_HZ = (4800.0, 6800.0)
GRAIN_SAMPLES = 1920  # ~80 ms burst, roughly one label frame
ATTACK_SAMPLES = 48  # ~2 ms sharp attack so a flux spike appears at the onset frame
_MIN_ONSET_GAP_FRAMES = 4  # keep collar windows (plus or minus 2 frames) unambiguous

REGIME_FAVORABLE = "favorable"
REGIME_NULL = "null"
REGIMES = (REGIME_FAVORABLE, REGIME_NULL)


@dataclass(frozen=True, slots=True)
class SyntheticStarssConfig:
    """Knobs for the synthetic corpus. Defaults are small and fast; scale up for a full run."""

    clip_seconds: float = 6.0
    onsets_per_clip: int = 8
    nuisance_per_clip: int = 8
    grain_amplitude: float = 0.6
    background_amplitude: float = 0.03
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.clip_seconds <= 0:
            raise ValueError("clip_seconds must be positive")
        if self.onsets_per_clip <= 0:
            raise ValueError("onsets_per_clip must be positive")
        if self.nuisance_per_clip < 0:
            raise ValueError("nuisance_per_clip must be nonnegative")

    @property
    def n_frames(self) -> int:
        return int(round(self.clip_seconds * 1000.0 / 100.0))

    @property
    def n_samples(self) -> int:
        return self.n_frames * SAMPLES_PER_FRAME


def _rng_for(base_seed: int, namespace: str) -> np.random.Generator:
    """Return a generator domain-separated by ``(base_seed, namespace)``.

    ``mop.seeding.derive_seed32`` passes an in-range seed through unchanged and ignores its namespace,
    so it cannot separate two rooms, clips, or roles that share a small integer seed (every namespace
    would collapse to the same stream at seed 0). The full key is therefore hashed here so each room,
    clip, and role gets its own byte-reproducible stream.
    """

    payload = json.dumps(
        {"base_seed": int(base_seed), "namespace": str(namespace)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    child = int.from_bytes(hashlib.sha256(b"mop-starss23-fixtures-v1\0" + payload).digest()[:4], "big")
    return np.random.default_rng(child)


def _band_grain(rng: np.random.Generator, band: tuple[float, float], amplitude: float) -> np.ndarray:
    """A short mono burst confined to a spectral band with a sharp attack (a flux-spike source)."""

    t = np.arange(GRAIN_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    n_tones = 4
    freqs = rng.uniform(band[0], band[1], size=n_tones)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_tones)
    carrier = np.zeros(GRAIN_SAMPLES, dtype=np.float64)
    for freq, phase in zip(freqs, phases, strict=True):
        carrier += np.sin(2.0 * np.pi * freq * t + phase)
    carrier /= n_tones
    envelope = np.exp(-t / 0.03)  # ~30 ms decay
    attack = np.minimum(1.0, np.arange(GRAIN_SAMPLES, dtype=np.float64) / ATTACK_SAMPLES)
    return amplitude * carrier * envelope * attack


def _ambisonics_gains(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """First-order Ambisonics channel gains [W, X, Y, Z] for one steered mono source."""

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    w = (1.0 / np.sqrt(2.0)) / distance
    x = (np.cos(az) * np.cos(el)) / distance
    y = (np.sin(az) * np.cos(el)) / distance
    z = np.sin(el) / distance
    return np.array([w, x, y, z], dtype=np.float64)


def _room_background(rng: np.random.Generator, room_id: str, n_samples: int, amplitude: float) -> np.ndarray:
    """Room-correlated colored background, shape (N_CHANNELS, n_samples). Deterministic per room."""

    room_rng = _rng_for(0, f"room-color:{room_id}")
    # A fixed per-room AR(1) coloring coefficient makes each room's spectrum distinct and correlated.
    color = float(room_rng.uniform(0.55, 0.95))
    white = rng.standard_normal((N_CHANNELS, n_samples))
    colored = np.empty_like(white)
    colored[:, 0] = white[:, 0]
    for i in range(1, n_samples):
        colored[:, i] = color * colored[:, i - 1] + np.sqrt(1.0 - color * color) * white[:, i]
    return amplitude * colored


def _place(audio: np.ndarray, start_sample: int, grain: np.ndarray, gains: np.ndarray) -> None:
    end = min(audio.shape[1], start_sample + grain.shape[0])
    span = end - start_sample
    if span <= 0:
        return
    audio[:, start_sample:end] += gains[:, None] * grain[None, :span]


def _draw_frames(rng: np.random.Generator, n_frames: int, count: int, forbidden: set[int]) -> list[int]:
    """Draw ``count`` well-separated frames avoiding ``forbidden`` and each other's collar."""

    chosen: list[int] = []
    taken = set(forbidden)
    attempts = 0
    limit = 2 + n_frames // _MIN_ONSET_GAP_FRAMES
    while len(chosen) < count and attempts < 200 * count:
        attempts += 1
        frame = int(rng.integers(1, n_frames - 1))
        if any(abs(frame - other) < _MIN_ONSET_GAP_FRAMES for other in taken):
            continue
        chosen.append(frame)
        taken.add(frame)
        if len(chosen) >= limit:
            break
    return sorted(chosen)


def generate_clip(
    *,
    clip_id: str,
    room_id: str,
    regime: str,
    config: SyntheticStarssConfig,
) -> tuple[Clip, np.ndarray]:
    """Generate one deterministic clip and its 4-channel audio for the given regime.

    favorable: onset grains sit at the labeled onset frames (spatial oracle scores F1 = 1.0), with
    nuisance grains in a disjoint band elsewhere.
    null: audio is nuisance-only and the onset labels are placed at independent frames that carry no
    grain, so no feature predicts them and a learned gate cannot beat rate-matched-random.
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    n_frames = config.n_frames
    n_samples = config.n_samples
    audio = _room_background(
        _rng_for(config.base_seed, f"bg:{clip_id}"), room_id, n_samples, config.background_amplitude
    )
    grain_rng = _rng_for(config.base_seed, f"grain:{clip_id}")
    doa_rng = _rng_for(config.base_seed, f"doa:{clip_id}")

    onsets: list[OnsetEvent] = []
    if regime == REGIME_FAVORABLE:
        onset_frames = _draw_frames(grain_rng, n_frames, config.onsets_per_clip, forbidden=set())
        for frame in onset_frames:
            # STARSS23 direction-of-arrival labels are integer degrees; distance is an integer count.
            az = float(doa_rng.integers(-180, 181))
            el = float(doa_rng.integers(-90, 91))
            dist = float(doa_rng.integers(1, 4))
            grain = _band_grain(grain_rng, SIGNAL_BAND_HZ, config.grain_amplitude)
            _place(audio, frame * SAMPLES_PER_FRAME, grain, _ambisonics_gains(az, el, dist))
            onsets.append(
                OnsetEvent(
                    frame=frame,
                    class_id=int(doa_rng.integers(0, N_CLASSES)),
                    azimuth=az,
                    elevation=el,
                    distance=dist,
                )
            )
        nuisance_frames = _draw_frames(
            grain_rng, n_frames, config.nuisance_per_clip, forbidden=set(onset_frames)
        )
    else:
        # Null: no onset grains. Nuisance fills the audio; labels are independent of the audio.
        nuisance_frames = _draw_frames(grain_rng, n_frames, config.nuisance_per_clip, forbidden=set())
        label_frames = _draw_frames(
            doa_rng, n_frames, config.onsets_per_clip, forbidden=set(nuisance_frames)
        )
        for frame in label_frames:
            onsets.append(
                OnsetEvent(
                    frame=frame,
                    class_id=int(doa_rng.integers(0, N_CLASSES)),
                    azimuth=float(doa_rng.integers(-180, 181)),
                    elevation=float(doa_rng.integers(-90, 91)),
                    distance=float(doa_rng.integers(1, 4)),
                )
            )

    for frame in nuisance_frames:
        az = float(doa_rng.uniform(-180.0, 180.0))
        el = float(doa_rng.uniform(-90.0, 90.0))
        dist = float(doa_rng.uniform(1.0, 3.0))
        grain = _band_grain(grain_rng, NUISANCE_BAND_HZ, config.grain_amplitude)
        _place(audio, frame * SAMPLES_PER_FRAME, grain, _ambisonics_gains(az, el, dist))

    onsets.sort(key=lambda onset: onset.frame)
    audio_sha256 = hashlib.sha256(np.ascontiguousarray(audio, dtype="<f8").tobytes()).hexdigest()
    clip = Clip(
        clip_id=clip_id,
        room_id=room_id,
        n_frames=n_frames,
        audio_sha256=audio_sha256,
        onsets=tuple(onsets),
    )
    return clip, audio


def pure_aleatoric_frames(seed: int, n_frames: int) -> np.ndarray:
    """A block of (n_frames, D_FEAT) pure-aleatoric feature rows with no reducible structure.

    Used by the noisy-TV control: an honest gate must fire on these at chance, never preferentially.
    """

    from .featurizer import D_FEAT

    rng = _rng_for(seed, "noisy-tv-aleatoric")
    return rng.standard_normal((n_frames, D_FEAT))
