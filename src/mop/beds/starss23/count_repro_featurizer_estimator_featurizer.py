"""Bias-independent reproduction (axis: featurizer_estimator), part A: a re-authored frozen count front-end.

This is a NET-NEW, ADDITIVE module. It edits no sealed count_* or onset module and no existing proof. It
exists to adversarially test whether the counting bed's first mechanics-ok signal (the trained gate reaching
strictly lower coasted-count-MAE than rate-matched-random at a matched re-estimation budget) survives when
the frozen front-end is swapped for an independently authored one with a different frequency warping, a
different bandwidth law, and a different change statistic. If the win is a fingerprint of the sealed 32-mel
log-mel-flux front-end, it dies here; if the gate keys on real concurrent-source count-change structure, it
survives the swap.

What differs from the sealed ``count_featurizer.FrozenCountFeaturizer``
----------------------------------------------------------------------
1. FILTERBANK: a 32-band gammatone filterbank on the ERB (equivalent-rectangular-bandwidth) scale, replacing
   the mel triangular filterbank. Center frequencies are spaced uniformly on the Glasberg and Moore ERB-rate
   scale between ``F_MIN`` and Nyquist, each with its own ERB bandwidth ``b = 1.019 * ERB(fc)``, and the
   per-bin weight is the fourth-order gammatone magnitude response ``(1 + ((f - fc)/b)^2)^(-2)``. This is a
   dense, physiologically-motivated response with genuinely different bandwidths from the sparse triangular
   mel filters, not a re-warp of the same triangles.
2. CHANGE STATISTIC: a relative spectral flux in the LINEAR band-energy domain, normalized by a causal
   per-band running average and bounded with a hyperbolic-tangent squash, replacing the half-wave difference
   of LOG-mel energy. Per band and short-time column the flux is ``rf = (B[c] - B[c-1]) / (R[c] + eps)`` with
   ``R[c]`` a strictly causal exponential moving average of the band energy over columns before ``c``; the
   positive polarity accumulates ``max(0, tanh(rf))`` and the negative polarity ``max(0, tanh(-rf))`` across
   the five columns of each 100 ms frame. This is a normalized novelty signal, not a log difference.

What is held IDENTICAL to the sealed front-end (so the frozen gate consumes it byte-unchanged)
----------------------------------------------------------------------------------------------
The output is exactly ``D_CFEAT = 256`` per frame in the identical ``[128 positive | 128 negative]`` block
layout (32 bands x 4 channels x 2 polarities), so the held-fixed ``CountGate`` and ``CountOnlineState`` (which
read the two 128-blocks for the positive and negative flux peaks and the D_IN = 264 gate width) consume it
with no change at all. The DSP grid (1024 window, 1024 n_fft, 480 hop, 5 columns per 100 ms frame) is the
frozen native grid.

Like the sealed front-end, this carries ZERO trained parameters (``n_params() == 0``): the Hann window, the
gammatone filterbank, and the running-average decay are deterministic functions of the sample rate, not
learned weights. Compute is charged analytically for a host-independent ledger. Its ``parameter_digest``
seals the window bytes, the filterbank bytes, and the change-statistic constants, and is provably distinct
from the sealed mel front-end's digest.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .featurizer import hann_window
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_REPRO_FE_FEATURIZER_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-featurizer/v1"

# The DSP grid, identical to the sealed front-end so the frozen gate width and block layout carry over.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_BANDS = 32  # 32 gammatone bands; two flux polarities across 4 channels give the 256 count features
N_POLARITY = 2  # positive (source-enter) and negative (source-leave) flux
D_CFEAT = N_BANDS * N_CHANNELS * N_POLARITY  # 256 per-frame count features, matching the sealed width
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# Fixed, preregistered DSP priors for the re-authored front-end. Hand-set, corpus-independent, not tuned on
# labels, exactly like the sealed featurizer's mel constants.
F_MIN = 50.0  # low edge of the gammatone bank in Hz; gammatone centers are placed above it on the ERB scale
GAMMATONE_ORDER = 4  # fourth-order gammatone; magnitude rolloff exponent is -ORDER/2 = -2
ERB_BANDWIDTH_FACTOR = 1.019  # b = 1.019 * ERB(fc), the standard gammatone equivalent bandwidth
REL_FLUX_DECAY = 0.2  # causal per-band running-average EMA step for the relative-flux normalizer
REL_FLUX_EPS = 1e-6  # floor on the running-average normalizer so silent bands do not divide by zero

# Analytic per-column-per-channel FLOP budget. Every constant is a fixed integer so the ledger is
# host-independent. The gammatone filterbank is DENSE (each band weights every rFFT bin), so its multiply is
# charged as the full 2 * N_BANDS * N_BINS multiply-add, unlike the sealed sparse triangular mel multiply.
FLOPS_WINDOW = WINDOW  # 1024 Hann taper multiplies
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_POWER = 3 * N_BINS  # 1539 real/imag square-and-add
FLOPS_GAMMATONE = 2 * N_BANDS * N_BINS  # 32832 dense gammatone multiply-add to linear band energy
FLOPS_REL_FLUX = 15 * N_BANDS  # 480 per-band diff, normalize, EMA update, tanh squash, dual rectify-accum
FLOPS_PER_COL_PER_CH = (
    FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER + FLOPS_GAMMATONE + FLOPS_REL_FLUX
)  # 87075
FLOPS_PER_FRAME_COUNT = FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_741_500


class CountReproFeaturizerRefusal(ValueError):
    """Raised when the re-authored count featurizer input violates the frozen FOA acquisition contract."""


def _hz_to_erb_rate(hz: np.ndarray) -> np.ndarray:
    """Glasberg and Moore ERB-rate (number of ERBs below hz). Monotone, so it defines the warping."""

    return 21.4 * np.log10(4.37e-3 * hz + 1.0)


def _erb_rate_to_hz(erb: np.ndarray) -> np.ndarray:
    return (np.power(10.0, erb / 21.4) - 1.0) / 4.37e-3


def _erb_hz(hz: np.ndarray) -> np.ndarray:
    """Glasberg and Moore equivalent rectangular bandwidth in Hz at center frequency hz."""

    return 24.7 * (4.37e-3 * hz + 1.0)


def _gammatone_filterbank(sample_rate: int) -> np.ndarray:
    """Return a fixed fourth-order gammatone filterbank, shape (N_BANDS, N_BINS), float64, zero-trained.

    Center frequencies are uniform on the ERB-rate scale from ``F_MIN`` to Nyquist. Each band's magnitude
    weight at rFFT bin frequency f is ``(1 + ((f - fc)/b)^2)^(-ORDER/2)`` with ``b = 1.019 * ERB(fc)``, the
    standard fourth-order gammatone equivalent-bandwidth magnitude response. Each band is peak-normalized to
    unit maximum weight so no band dominates by its raw gain. Deterministic in the sample rate alone.
    """

    f_max = sample_rate / 2.0
    erb_lo = _hz_to_erb_rate(np.array([F_MIN]))[0]
    erb_hi = _hz_to_erb_rate(np.array([f_max]))[0]
    centers = _erb_rate_to_hz(np.linspace(erb_lo, erb_hi, N_BANDS))
    bandwidths = ERB_BANDWIDTH_FACTOR * _erb_hz(centers)
    bin_freqs = np.linspace(0.0, f_max, N_BINS)
    filters = np.zeros((N_BANDS, N_BINS), dtype=np.float64)
    exponent = -GAMMATONE_ORDER / 2.0
    for m in range(N_BANDS):
        ratio = (bin_freqs - centers[m]) / bandwidths[m]
        weights = np.power(1.0 + ratio * ratio, exponent)
        peak = float(weights.max())
        if peak > 0.0:
            weights = weights / peak
        filters[m] = weights
    return filters


@dataclass(frozen=True, slots=True)
class ReproCountFeaturizer(FrozenFeatureProvider):
    """The re-authored frozen count front-end: gammatone ERB bands plus causal relative spectral flux."""

    _flops_per_frame = FLOPS_PER_FRAME_COUNT
    _frame_count_refusal = CountReproFeaturizerRefusal
    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def filterbank(self) -> np.ndarray:
        return _gammatone_filterbank(self.sample_rate)

    def parameter_digest(self) -> str:
        """Digest of the fixed window, gammatone filterbank, and change-statistic constants."""

        payload = {
            "schema": COUNT_REPRO_FE_FEATURIZER_SCHEMA,
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "filterbank_sha256": hashlib.sha256(self.filterbank.astype("<f8").tobytes()).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_bands": N_BANDS,
            "n_polarity": N_POLARITY,
            "d_cfeat": D_CFEAT,
            "f_min": F_MIN,
            "gammatone_order": GAMMATONE_ORDER,
            "erb_bandwidth_factor": ERB_BANDWIDTH_FACTOR,
            "rel_flux_decay": REL_FLUX_DECAY,
            "rel_flux_eps": REL_FLUX_EPS,
            "sample_rate": self.sample_rate,
            "change_statistic": (
                "causal-relative-spectral-flux: rf=(B[c]-B[c-1])/(EMA_causal(B)+eps), tanh-squashed, "
                "positive and negative polarities summed over the columns of each frame"
            ),
        }
        return canonical_sha256(payload)

    def _columns(self, signal: np.ndarray, n_frames: int) -> np.ndarray:
        """Return the (n_cols, WINDOW) windowed short-time columns at the frozen hop. Byte-deterministic."""

        padded = np.zeros(n_frames * SAMPLES_PER_FRAME + PAD_RIGHT, dtype=np.float64)
        padded[: signal.shape[0]] = signal
        n_cols = n_frames * COLS_PER_FRAME
        windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP]
        windows = windows[:n_cols]
        return np.ascontiguousarray(windows) * self.window

    def _channel_flux(self, signal: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Positive and negative causal relative-flux for one channel: each (n_frames, N_BANDS) float64."""

        columns = self._columns(signal, n_frames)
        n_cols = columns.shape[0]
        spectra = np.fft.rfft(columns, n=N_FFT, axis=1)
        power = (spectra.real * spectra.real) + (spectra.imag * spectra.imag)  # (n_cols, N_BINS)
        band_energy = power @ self.filterbank.T  # (n_cols, N_BANDS) linear gammatone band energy

        # Strictly causal per-band running average: R[c] uses only band energies at columns before c.
        running = np.empty_like(band_energy)
        avg = np.zeros(N_BANDS, dtype=np.float64)
        for c in range(n_cols):
            running[c] = avg
            avg = (1.0 - REL_FLUX_DECAY) * avg + REL_FLUX_DECAY * band_energy[c]

        diff = np.zeros_like(band_energy)
        diff[1:] = band_energy[1:] - band_energy[:-1]
        rel = np.tanh(diff / (running + REL_FLUX_EPS))
        pos = np.maximum(0.0, rel).reshape(n_frames, COLS_PER_FRAME, N_BANDS).sum(axis=1)
        neg = np.maximum(0.0, -rel).reshape(n_frames, COLS_PER_FRAME, N_BANDS).sum(axis=1)
        return pos, neg

    def featurize(self, audio: np.ndarray) -> np.ndarray:
        """Featurize a ``(N_CHANNELS, n_samples)`` FOA array into ``(n_frames, D_CFEAT=256)`` float64.

        The 256-vector is ``[pos_flux(4 channels x 32 bands), neg_flux(4 channels x 32 bands)]``: the first
        128 columns are the source-enter (positive) polarity and the last 128 the source-leave (negative)
        polarity, byte-identical in layout to the sealed front-end so the frozen gate reads it unchanged.
        """

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise CountReproFeaturizerRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise CountReproFeaturizerRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        pos_blocks: list[np.ndarray] = []
        neg_blocks: list[np.ndarray] = []
        for ch in range(N_CHANNELS):
            pos, neg = self._channel_flux(audio[ch], n_frames)
            pos_blocks.append(pos)
            neg_blocks.append(neg)
        return np.concatenate([*pos_blocks, *neg_blocks], axis=1)
