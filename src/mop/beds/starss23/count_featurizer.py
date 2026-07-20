
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_FEATURIZER_SCHEMA = "mop-starss23-count-featurizer/v1"

WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_MEL = 32  # 32-mel filterbank; two flux polarities per band give 256 features across 4 channels
N_POLARITY = 2  # positive (source-enter) and negative (source-leave) flux
D_CFEAT = N_MEL * N_CHANNELS * N_POLARITY  # 256 per-frame count features
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

FLOPS_WINDOW = WINDOW  # 1024 Hann taper multiplies
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_POWER = 3 * N_BINS  # 1539 real/imag square-and-add
FLOPS_MEL = 2 * WINDOW  # 2048 sparse mel multiply-add
FLOPS_LOG = N_MEL  # 32 logarithms
FLOPS_POS_FLUX = 3 * N_MEL  # 96 subtract, rectify, accumulate for the positive polarity
FLOPS_NEG_FLUX = 3 * N_MEL  # 96 subtract, rectify, accumulate for the negative polarity
FLOPS_PER_COL_PER_CH = (
    FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_MEL + FLOPS_LOG + FLOPS_POS_FLUX + FLOPS_NEG_FLUX
)  # 56035
FLOPS_PER_FRAME_COUNT = FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_120_700

_LOG_EPS = 1e-6


class CountFeaturizerRefusal(ValueError):
    pass


def hann_window() -> np.ndarray:
    n = np.arange(WINDOW, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WINDOW)


def mel_filterbank(sample_rate: int, n_mel: int = N_MEL, n_bins: int = N_BINS) -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)
    f_max = sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(f_max), n_mel + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.linspace(0.0, f_max, n_bins)
    filters = np.zeros((n_mel, n_bins), dtype=np.float64)
    for m in range(1, n_mel + 1):
        left, center, right = hz_points[m - 1 : m + 2]
        rising = (bin_freqs - left) / max(center - left, np.finfo(float).eps)
        falling = (right - bin_freqs) / max(right - center, np.finfo(float).eps)
        filters[m - 1] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


@dataclass(frozen=True, slots=True)
class FrozenCountFeaturizer(FrozenFeatureProvider):

    _flops_per_frame = FLOPS_PER_FRAME_COUNT
    _frame_count_refusal = CountFeaturizerRefusal
    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def filterbank(self) -> np.ndarray:
        return mel_filterbank(self.sample_rate, N_MEL, N_BINS)

    def parameter_digest(self) -> str:

        payload = {
            "schema": COUNT_FEATURIZER_SCHEMA,
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "filterbank_sha256": hashlib.sha256(self.filterbank.astype("<f8").tobytes()).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_mel": N_MEL,
            "n_polarity": N_POLARITY,
            "d_cfeat": D_CFEAT,
            "sample_rate": self.sample_rate,
        }
        return canonical_sha256(payload)

    def _channel_flux(self, signal: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:

        padded = np.zeros(n_frames * SAMPLES_PER_FRAME + PAD_RIGHT, dtype=np.float64)
        padded[: signal.shape[0]] = signal
        n_cols = n_frames * COLS_PER_FRAME
        window = self.window
        filterbank = self.filterbank
        logmel = np.empty((n_cols, N_MEL), dtype=np.float64)
        for c in range(n_cols):
            start = c * HOP
            frame = padded[start : start + WINDOW] * window
            spectrum = np.fft.rfft(frame, n=N_FFT)
            power = (spectrum.real * spectrum.real) + (spectrum.imag * spectrum.imag)
            mel = filterbank @ power
            logmel[c] = np.log(mel + _LOG_EPS)
        diff = np.zeros_like(logmel)
        diff[1:] = logmel[1:] - logmel[:-1]
        pos = np.maximum(0.0, diff).reshape(n_frames, COLS_PER_FRAME, N_MEL).sum(axis=1)
        neg = np.maximum(0.0, -diff).reshape(n_frames, COLS_PER_FRAME, N_MEL).sum(axis=1)
        return pos, neg

    def featurize(self, audio: np.ndarray) -> np.ndarray:

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise CountFeaturizerRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise CountFeaturizerRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        pos_blocks: list[np.ndarray] = []
        neg_blocks: list[np.ndarray] = []
        for ch in range(N_CHANNELS):
            pos, neg = self._channel_flux(audio[ch], n_frames)
            pos_blocks.append(pos)
            neg_blocks.append(neg)
        return np.concatenate([*pos_blocks, *neg_blocks], axis=1)
