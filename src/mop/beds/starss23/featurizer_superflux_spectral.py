"""A net-new FROZEN zero-trained-parameter SuperFlux onset front-end for the STARSS23 ESCS bed.

This is an additive component. It changes no sealed scoring path. It implements the same featurizer
interface as the committed ``FrozenFeaturizer`` (``featurize``, ``n_params``, ``parameter_digest``,
``feature_digest``, ``flops_for_frames`` plus a module-level analytic ``FLOPS_PER_FRAME``) but swaps the
front-end DSP for a frozen SuperFlux onset detector (Boeck and Widmer, "Maximum Filter Vibrato
Suppression for Onset Detection", DAFx 2013).

Mechanism (all fixed DSP, zero trained parameters)
--------------------------------------------------
Per short-time column, per FOA channel:

1. Hann window (1024) then n_fft 1024 real FFT (513 one-sided bins), identical grid to the base
   front-end, so five STFT columns at hop 480 tile exactly one 100 ms / 2400-sample label frame and the
   output tiles ``COLS_PER_FRAME * n_frames`` columns.
2. Power spectrum projected through the same fixed triangular 64-mel filterbank.
3. A mu-law-style logarithmic magnitude companding ``log1p(MU * mel) / log1p(MU)`` with a fixed MU. This
   is the frozen loudness compression; it carries no learned quantity.
4. The SuperFlux core: a maximum filter of a fixed small radius is applied ACROSS FREQUENCY to the
   PREVIOUS column's compressed magnitudes before the positive spectral difference is taken. The
   frequency max filter tracks a partial's short trajectory across a small band, so a vibrato or a
   fundamental wobble of up to the filter radius does not register as an onset. The per-bin novelty is
   the half-wave-rectified difference ``max(0, comp[t] - freqmax(comp[t - 1]))``.
5. The five columns of each 100 ms frame are summed, giving 64 mel-flux features per channel; the four
   channels concatenate to exactly 256 per-frame features, so the unchanged 256-input gate consumes the
   output with no projection, truncation, or padding, and adds no trained parameter.

Compute is charged analytically, never measured, so the ledger is reproducible across hosts. The
per-column-per-channel FLOP budget mirrors the base ledger with the mu-law companding replacing the plain
log and the frequency max filter added, so this front-end is slightly costlier per frame than the base
and that higher cost is charged honestly to every arm.

Zero trained parameters is asserted two ways, exactly as the base front-end asserts it: ``n_params()``
returns 0, and ``parameter_digest()`` hashes only deterministic functions of the sample rate, the n_fft,
and the fixed DSP constants (the Hann window, the mel filterbank, MU, and the max-filter radius), never a
learned weight. The output is byte-reproducible: identical input bytes yield identical output bytes.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

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


def _hann_window() -> np.ndarray:
    """Return the fixed periodic Hann window of length WINDOW as float64. Zero trained parameters."""

    n = np.arange(WINDOW, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WINDOW)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    """Return a fixed triangular 64-mel filterbank, shape (N_MEL, N_BINS), float64, zero-trained."""

    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(np.array([f_min]))[0], _hz_to_mel(np.array([f_max]))[0], N_MEL + 2)
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


def _frequency_max_filter(comp: np.ndarray, radius: int) -> np.ndarray:
    """Edge-clamped maximum filter of the given radius along the frequency axis (axis 1).

    ``comp`` is (T, N_MEL). For each mel bin the output is the maximum over the ``2 * radius + 1`` band
    centered on it, with the band clamped (edge-replicated) at the two extremes. This is the SuperFlux
    frequency max filter: a partial that wobbles across up to ``radius`` bins between adjacent columns
    stays inside the band and so does not register as an onset. Deterministic and byte-reproducible.
    """

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
class SuperfluxSpectralFeaturizer:
    """The frozen SuperFlux onset front-end. Deep-frozen: window, filterbank, MU, radius are fixed DSP."""

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
        """Digest of the fixed DSP constants, proving the SuperFlux front-end is byte-frozen.

        Hashes only deterministic functions of the sample rate and n_fft (the Hann window and the mel
        filterbank) plus the fixed SuperFlux constants MU and the max-filter radius. No learned weight
        enters, so ``n_params()`` is 0 and this digest is a pure function of the frozen configuration. It
        differs from the base front-end's digest because MU and the frequency max filter are new fixed
        DSP, so a feature cache keyed on it can never be served the base front-end's bytes and vice versa.
        """

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

    def flops_for_frames(self, n_frames: int) -> int:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer")
        return FLOPS_PER_FRAME * n_frames

    def _channel_superflux(self, signal: np.ndarray, n_frames: int) -> np.ndarray:
        """SuperFlux novelty for one channel: returns (n_frames, N_MEL) float64.

        Per column: window, rFFT, power, mel projection, mu-law companding. Then the SuperFlux positive
        difference against the frequency-max-filtered previous column, half-wave rectified. The
        COLS_PER_FRAME columns of each label frame are summed.
        """

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
        per_channel = [self._channel_superflux(audio[ch], n_frames) for ch in range(N_CHANNELS)]
        return np.concatenate(per_channel, axis=1)

    def feature_digest(self, features: np.ndarray) -> str:
        """Digest of a feature block, used to assert byte-reproducibility of the front-end."""

        return hashlib.sha256(np.ascontiguousarray(features, dtype="<f8").tobytes()).hexdigest()
