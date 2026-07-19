"""Component 2: the frozen zero-trained-parameter deterministic DSP front-end.

The featurizer is byte-reproducible and carries no trained parameter. Its per-frame output is the
half-wave-rectified log-mel spectral flux of each of the four FOA channels, dim D = 256 (64 mel by 4
channels). The pipeline per short-time column is: Hann window (1024), n_fft 1024 real FFT (513 bins),
power, a fixed sparse 64-mel projection, log, then a half-wave-rectified flux across columns. Five
STFT columns at hop 480 tile exactly one 100 ms / 2400-sample label frame.

Compute is charged analytically, not measured. The per-column-per-channel cost is a fixed constant so
the ledger is reproducible across hosts. n_params() is exactly zero: the Hann window and the mel
filterbank are deterministic functions of the sample rate and n_fft, not learned weights.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .schema import N_CHANNELS, SAMPLES_PER_FRAME

# The DSP grid. The front-end owns these; the schema owns only the label grid.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_MEL = 64
D_FEAT = N_MEL * N_CHANNELS  # 256 per-frame features
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# Analytic per-column-per-channel FLOP budget (docs/ESCS_DEEP_RESEARCH.md immediate design implications).
FLOPS_WINDOW = WINDOW  # 1024 multiplies for the Hann taper
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_POWER = 3 * N_BINS  # 1539 real/imag square-and-add
FLOPS_MEL = 2 * WINDOW  # 2048 sparse mel multiply-add (nnz ~ 1024)
FLOPS_LOG = N_MEL  # 64 logarithms
FLOPS_FLUX = 3 * N_MEL  # 192 subtract, rectify, accumulate
FLOPS_PER_COL_PER_CH = (
    FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_MEL + FLOPS_LOG + FLOPS_FLUX
)  # 56067
FLOPS_PER_FRAME = FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_121_340

_LOG_EPS = 1e-6


def hann_window() -> np.ndarray:
    """Return the fixed periodic Hann window of length WINDOW as float64. Zero trained parameters."""

    n = np.arange(WINDOW, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WINDOW)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_mel: int = N_MEL, n_bins: int = N_BINS) -> np.ndarray:
    """Return one fixed triangular mel filterbank as a float64 zero-trained DSP matrix."""

    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(np.array([f_min]))[0], hz_to_mel(np.array([f_max]))[0], n_mel + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.linspace(0.0, f_max, n_bins)
    filters = np.zeros((n_mel, n_bins), dtype=np.float64)
    for m in range(1, n_mel + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        for b in range(n_bins):
            freq = bin_freqs[b]
            if left <= freq <= center and center > left:
                filters[m - 1, b] = (freq - left) / (center - left)
            elif center <= freq <= right and right > center:
                filters[m - 1, b] = (right - freq) / (right - center)
    return filters


@dataclass(frozen=True, slots=True)
class FrozenFeaturizer:
    """The frozen deterministic front-end. Deep-frozen: window and filterbank are fixed DSP, not weights."""

    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def filterbank(self) -> np.ndarray:
        return mel_filterbank(self.sample_rate)

    def n_params(self) -> int:
        """Zero trained parameters. The front-end is a deterministic DSP, never a learned encoder."""

        return 0

    def parameter_digest(self) -> str:
        """Digest of the fixed window and filterbank bytes, proving the front-end is byte-frozen."""

        payload = {
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "filterbank_sha256": hashlib.sha256(self.filterbank.astype("<f8").tobytes()).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_mel": N_MEL,
            "sample_rate": self.sample_rate,
        }
        return canonical_sha256(payload)

    def flops_for_frames(self, n_frames: int) -> int:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer")
        return FLOPS_PER_FRAME * n_frames

    def _channel_flux(self, signal: np.ndarray, n_frames: int) -> np.ndarray:
        """Half-wave-rectified log-mel flux for one channel: returns (n_frames, N_MEL) float64."""

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
        # Half-wave-rectified spectral flux across columns; flux at the very first column is zero.
        flux = np.zeros_like(logmel)
        flux[1:] = np.maximum(0.0, logmel[1:] - logmel[:-1])
        # Aggregate the COLS_PER_FRAME columns of each label frame by summation.
        return flux.reshape(n_frames, COLS_PER_FRAME, N_MEL).sum(axis=1)

    def featurize(self, audio: np.ndarray) -> np.ndarray:
        """Featurize a (N_CHANNELS, n_samples) FOA array into (n_frames, D_FEAT=256) float64.

        Byte-reproducible: identical input bytes yield identical output bytes on a given host.
        """

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise ValueError(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise ValueError("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        per_channel = [self._channel_flux(audio[ch], n_frames) for ch in range(N_CHANNELS)]
        return np.concatenate(per_channel, axis=1)

    def feature_digest(self, features: np.ndarray) -> str:
        """Digest of a feature block, used to assert byte-reproducibility of the front-end."""

        return hashlib.sha256(np.ascontiguousarray(features, dtype="<f8").tobytes()).hexdigest()
