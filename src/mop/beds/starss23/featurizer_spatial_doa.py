
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .featurizer import hann_window
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

N_BANDS = 64
FEATURES_PER_BAND = 4
D_FEAT = N_BANDS * FEATURES_PER_BAND  # 256 per-frame features; matches the gate's hardcoded contract

ACN_W, ACN_Y, ACN_Z, ACN_X = 0, 1, 2, 3

_EPS = 1e-12


FLOPS_WINDOW = WINDOW  # 1024 multiplies for the Hann taper
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_STFT_PER_COL_PER_CH = FLOPS_WINDOW + FLOPS_RFFT  # 52224
FLOPS_STFT = FLOPS_STFT_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_044_480

FLOPS_INTENSITY_PER_COL_PER_BIN = 9 + 12 + 4 + 4  # 29
FLOPS_INTENSITY = FLOPS_INTENSITY_PER_COL_PER_BIN * COLS_PER_FRAME * N_BINS  # 29 * 5 * 513 = 74_385

FLOPS_REDUCE_PER_BAND = 13 + 3 + 10 + 11 + 10 + 32 + 2 + 2  # 83
FLOPS_REDUCE = FLOPS_REDUCE_PER_BAND * N_BANDS  # 83 * 64 = 5_312

FLOPS_PER_FRAME = FLOPS_STFT + FLOPS_INTENSITY + FLOPS_REDUCE  # 1_044_480 + 74_385 + 5_312 = 1_124_177


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * float(np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (float(np.power(10.0, mel / 2595.0)) - 1.0)


def _band_edges(sample_rate: int) -> np.ndarray:

    f_max = sample_rate / 2.0
    mel_lo, mel_hi = _hz_to_mel(0.0), _hz_to_mel(f_max)
    mel_points = np.linspace(mel_lo, mel_hi, N_BANDS + 1)
    hz_points = np.array([_mel_to_hz(float(m)) for m in mel_points], dtype=np.float64)
    raw = np.round(hz_points / f_max * (N_BINS - 1)).astype(np.int64)
    edges = np.empty(N_BANDS + 1, dtype=np.int64)
    edges[0] = 0
    edges[N_BANDS] = N_BINS
    for b in range(1, N_BANDS):
        lo = edges[b - 1] + 1
        hi = N_BINS - (N_BANDS - b)
        edges[b] = int(min(max(int(raw[b]), lo), hi))
    return edges


def _band_membership(sample_rate: int) -> np.ndarray:

    edges = _band_edges(sample_rate)
    membership = np.zeros((N_BANDS, N_BINS), dtype=np.float64)
    for b in range(N_BANDS):
        membership[b, edges[b] : edges[b + 1]] = 1.0
    return membership


@dataclass(frozen=True, slots=True)
class SpatialDoaFeaturizer(FrozenFeatureProvider):

    _flops_per_frame = FLOPS_PER_FRAME
    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def band_membership(self) -> np.ndarray:
        return _band_membership(self.sample_rate)

    def parameter_digest(self) -> str:

        membership = self.band_membership
        payload = {
            "front_end": "spatial_doa_active_intensity",
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "band_membership_sha256": hashlib.sha256(membership.astype("<f8").tobytes()).hexdigest(),
            "band_edges": [int(v) for v in _band_edges(self.sample_rate)],
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_bands": N_BANDS,
            "features_per_band": FEATURES_PER_BAND,
            "acn_channel_map": {"W": ACN_W, "Y": ACN_Y, "Z": ACN_Z, "X": ACN_X},
            "sample_rate": self.sample_rate,
        }
        return canonical_sha256(payload)

    def _channel_spectra(self, signal: np.ndarray, n_cols: int) -> np.ndarray:

        padded = np.zeros(n_cols * HOP + PAD_RIGHT, dtype=np.float64)
        padded[: signal.shape[0]] = signal
        window = self.window
        columns = np.empty((n_cols, WINDOW), dtype=np.float64)
        for c in range(n_cols):
            start = c * HOP
            columns[c] = padded[start : start + WINDOW]
        columns *= window
        return np.fft.rfft(columns, n=N_FFT, axis=1)

    def featurize(self, audio: np.ndarray) -> np.ndarray:

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise ValueError(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise ValueError("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        n_cols = n_frames * COLS_PER_FRAME

        spectra = [self._channel_spectra(audio[ch], n_cols) for ch in range(N_CHANNELS)]
        w = spectra[ACN_W]
        x = spectra[ACN_X]
        y = spectra[ACN_Y]
        z = spectra[ACN_Z]

        ix = w.real * x.real + w.imag * x.imag
        iy = w.real * y.real + w.imag * y.imag
        iz = w.real * z.real + w.imag * z.imag
        energy = 0.5 * (
            (w.real * w.real + w.imag * w.imag)
            + (x.real * x.real + x.imag * x.imag)
            + (y.real * y.real + y.imag * y.imag)
            + (z.real * z.real + z.imag * z.imag)
        )

        membership = self.band_membership  # (N_BANDS, N_BINS)

        def _band_sum(field: np.ndarray) -> np.ndarray:
            per_frame = field.reshape(n_frames, COLS_PER_FRAME, N_BINS).sum(axis=1)
            return per_frame @ membership.T  # (n_frames, N_BANDS)

        sum_ix = _band_sum(ix)
        sum_iy = _band_sum(iy)
        sum_iz = _band_sum(iz)
        sum_e = _band_sum(energy)

        intensity_norm = np.sqrt(sum_ix * sum_ix + sum_iy * sum_iy + sum_iz * sum_iz)
        diffuseness = np.clip(1.0 - intensity_norm / (sum_e + _EPS), 0.0, 1.0)

        vx, vy, vz = -sum_ix, -sum_iy, -sum_iz
        azimuth = np.arctan2(vy, vx)
        horizontal = np.sqrt(vx * vx + vy * vy)
        elevation = np.arctan2(vz, horizontal)
        cos_el = np.cos(elevation)
        dir_x = cos_el * np.cos(azimuth)
        dir_y = cos_el * np.sin(azimuth)
        dir_z = np.sin(elevation)

        features = np.concatenate([dir_x, dir_y, dir_z, diffuseness], axis=1)
        return np.ascontiguousarray(features, dtype=np.float64)
