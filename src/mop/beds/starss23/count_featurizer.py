"""Concurrent-source-counting bed, component 2: the frozen zero-parameter count front-end.

The count featurizer is byte-reproducible and carries no trained parameter, exactly like the sealed onset
featurizer. Its per-frame output is a 256-vector built from a 32-mel log-mel spectrogram of each of the
four FOA channels, split into a positive (source-enter) flux readout and a negative (source-leave) flux
readout across short-time columns:

    per (mel band, channel): pos_flux = sum_cols max(0, logmel[c] - logmel[c-1])   (32 x 4 = 128 features)
                             neg_flux = sum_cols max(0, logmel[c-1] - logmel[c])   (32 x 4 = 128 features)
    D_CFEAT = 128 + 128 = 256

A concurrent-source COUNT changes exactly when a source enters (a positive log-mel attack) or leaves (a
negative log-mel decay), so a change-detector gate reads its evidence from these two flux polarities. The
256-vector keeps the gate input width at D_IN = 256 + 8 = 264, byte-identical to the sealed onset gate, so
the reused parameter and FLOP anchors of gate.py carry over unchanged.

Compute is charged analytically, not measured, so the ledger is reproducible across hosts. n_params() is
exactly zero: the Hann window and the 32-mel filterbank are deterministic functions of the sample rate
and n_fft, not learned weights.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_FEATURIZER_SCHEMA = "mop-starss23-count-featurizer/v1"

# The DSP grid. The count front-end owns these; the schema owns only the label grid.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_MEL = 32  # 32-mel filterbank; two flux polarities per band give 256 features across 4 channels
N_POLARITY = 2  # positive (source-enter) and negative (source-leave) flux
D_CFEAT = N_MEL * N_CHANNELS * N_POLARITY  # 256 per-frame count features
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# Analytic per-column-per-channel FLOP budget. The base flux DSP mirrors the onset front-end column cost;
# the count front-end adds the second (negative) flux polarity. Every constant is a fixed integer so the
# ledger is host-independent.
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
    """Raised when the count featurizer input violates the frozen FOA acquisition contract."""


def _hann_window() -> np.ndarray:
    n = np.arange(WINDOW, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WINDOW)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    """Return a fixed triangular 32-mel filterbank, shape (N_MEL, N_BINS), float64, zero-trained."""

    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_points = np.linspace(
        _hz_to_mel(np.array([f_min]))[0], _hz_to_mel(np.array([f_max]))[0], N_MEL + 2
    )
    hz_points = _mel_to_hz(mel_points)
    bin_freqs = np.linspace(0.0, f_max, N_BINS)
    filters = np.zeros((N_MEL, N_BINS), dtype=np.float64)
    for m in range(1, N_MEL + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        for b in range(N_BINS):
            freq = bin_freqs[b]
            if left <= freq <= center and center > left:
                filters[m - 1, b] = (freq - left) / (center - left)
            elif center <= freq <= right and right > center:
                filters[m - 1, b] = (right - freq) / (right - center)
    return filters


@dataclass(frozen=True, slots=True)
class FrozenCountFeaturizer:
    """The frozen deterministic count front-end. Window and 32-mel filterbank are fixed DSP, not weights."""

    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return _hann_window()

    @property
    def filterbank(self) -> np.ndarray:
        return _mel_filterbank(self.sample_rate)

    def n_params(self) -> int:
        """Zero trained parameters. The front-end is a deterministic DSP, never a learned encoder."""

        return 0

    def parameter_digest(self) -> str:
        """Digest of the fixed window and filterbank bytes, proving the front-end is byte-frozen."""

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

    def flops_for_frames(self, n_frames: int) -> int:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise CountFeaturizerRefusal("n_frames must be a nonnegative integer")
        return FLOPS_PER_FRAME_COUNT * n_frames

    def _channel_flux(self, signal: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Positive and negative half-wave log-mel flux for one channel: each (n_frames, N_MEL) float64."""

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
        """Featurize a ``(N_CHANNELS, n_samples)`` FOA array into ``(n_frames, D_CFEAT=256)`` float64.

        The 256-vector is ``[pos_flux(4 channels x 32 mel), neg_flux(4 channels x 32 mel)]``: the first
        128 columns are the source-enter (positive) polarity and the last 128 are the source-leave
        (negative) polarity. Byte-reproducible for identical input bytes on a given host.
        """

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

    def feature_digest(self, features: np.ndarray) -> str:
        """Digest of a feature block, used to assert byte-reproducibility of the front-end."""

        return hashlib.sha256(np.ascontiguousarray(features, dtype="<f8").tobytes()).hexdigest()
