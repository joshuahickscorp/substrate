
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .featurizer import hann_window, mel_filterbank
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

# The DSP grid. Identical to the base front-end so the label-frame tiling and the 256-dim output match.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_MEL = 64
D_FEAT = N_MEL * N_CHANNELS  # 256 per-frame features (matches the gate's hard-wired D_FEAT)
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# SuperFlux front-end constants. Both are fixed DSP, not learned: a mu-law-style companding constant and
# the maximum-filter radius in mel bins that suppresses vibrato / partial wobble across frequency.
MU = 1000.0  # mu-law-style logarithmic magnitude companding constant
LOG1P_MU = math.log1p(MU)  # normalizer so a unit mel maps into a bounded compressed magnitude
MAX_FILTER_RADIUS = 1  # frequency max-filter half-width in mel bins (a 2R+1 = 3-tap band)

# Analytic per-column-per-channel FLOP budget. Mirrors the base ledger, with the mu-law companding
# replacing the plain log (FLOPS_LOG -> FLOPS_COMPRESS) and the frequency max filter added
# (FLOPS_MAXFILT). The values are host-independent so the FLOP ledger is byte-reproducible.
FLOPS_WINDOW = WINDOW  # 1024 multiplies for the Hann taper
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_POWER = 3 * N_BINS  # 1539 real/imag square-and-add
FLOPS_MEL = 2 * WINDOW  # 2048 sparse mel multiply-add (nnz ~ 1024)
FLOPS_COMPRESS = 4 * N_MEL  # 256 mu-law companding: multiply by MU, log1p, add, divide, per mel bin
FLOPS_MAXFILT = (2 * MAX_FILTER_RADIUS + 1) * N_MEL  # 192 max over the (2R+1)-tap frequency band per bin
FLOPS_FLUX = 3 * N_MEL  # 192 subtract, rectify, accumulate
FLOPS_PER_COL_PER_CH = (
    FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_MEL + FLOPS_COMPRESS + FLOPS_MAXFILT + FLOPS_FLUX
)  # 56451
FLOPS_PER_FRAME = FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_129_020


def _frequency_max_filter(comp: np.ndarray, radius: int) -> np.ndarray:

    if radius <= 0:
        return comp
    out = comp.copy()
    for shift in range(1, radius + 1):
        left = np.empty_like(comp)
        left[:, shift:] = comp[:, :-shift]
        left[:, :shift] = comp[:, :1]
        right = np.empty_like(comp)
        right[:, :-shift] = comp[:, shift:]
        right[:, -shift:] = comp[:, -1:]
        out = np.maximum(out, np.maximum(left, right))
    return out


@dataclass(frozen=True, slots=True)
class SuperfluxSpectralFeaturizer(FrozenFeatureProvider):

    _flops_per_frame = FLOPS_PER_FRAME
    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def filterbank(self) -> np.ndarray:
        return mel_filterbank(self.sample_rate)

    def parameter_digest(self) -> str:

        payload = {
            "front_end": "superflux_spectral",
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "filterbank_sha256": hashlib.sha256(self.filterbank.astype("<f8").tobytes()).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_mel": N_MEL,
            "mu": MU,
            "max_filter_radius": MAX_FILTER_RADIUS,
            "sample_rate": self.sample_rate,
        }
        return canonical_sha256(payload)

    def _channel_superflux(self, signal: np.ndarray, n_frames: int) -> np.ndarray:

        padded = np.zeros(n_frames * SAMPLES_PER_FRAME + PAD_RIGHT, dtype=np.float64)
        padded[: signal.shape[0]] = signal
        n_cols = n_frames * COLS_PER_FRAME
        window = self.window
        filterbank = self.filterbank
        comp = np.empty((n_cols, N_MEL), dtype=np.float64)
        for c in range(n_cols):
            start = c * HOP
            frame = padded[start : start + WINDOW] * window
            spectrum = np.fft.rfft(frame, n=N_FFT)
            power = (spectrum.real * spectrum.real) + (spectrum.imag * spectrum.imag)
            mel = filterbank @ power
            # Mu-law-style logarithmic magnitude companding (fixed DSP, zero trained parameters).
            comp[c] = np.log1p(MU * mel) / LOG1P_MU
        # SuperFlux: half-wave-rectified positive difference against the frequency-max-filtered previous
        # column. The novelty at the very first column is zero (no previous column to difference).
        flux = np.zeros_like(comp)
        if n_cols > 1:
            prev_freqmax = _frequency_max_filter(comp[:-1], MAX_FILTER_RADIUS)
            flux[1:] = np.maximum(0.0, comp[1:] - prev_freqmax)
        # Aggregate the COLS_PER_FRAME columns of each label frame by summation.
        return flux.reshape(n_frames, COLS_PER_FRAME, N_MEL).sum(axis=1)

    def featurize(self, audio: np.ndarray) -> np.ndarray:

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise ValueError(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise ValueError("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        per_channel = [self._channel_superflux(audio[ch], n_frames) for ch in range(N_CHANNELS)]
        return np.concatenate(per_channel, axis=1)
