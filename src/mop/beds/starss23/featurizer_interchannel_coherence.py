
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .featurizer import hann_window, hz_to_mel
from .schema import N_CHANNELS, SAMPLE_RATE_HZ, SAMPLES_PER_FRAME

# The DSP grid. The front-end owns these; the schema owns only the label grid. The STFT grid is the same
# as the committed featurizer so the two front-ends tile a 100 ms label frame identically.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
N_BANDS = 64  # mel-spaced coherence bands
N_SPATIAL = 4  # MSC(W,X), MSC(W,Y), MSC(W,Z), directness
D_FEAT = N_BANDS * N_SPATIAL  # 256 per-frame features; the sealed gate consumes exactly 256
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# Numerical floor for the normalized ratios. A silent band yields zero coherence and zero directness.
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Analytic per-frame FLOP ledger. Every count is host-independent, so the ledger is reproducible and can
# be charged identically to every arm by the matched-budget harness.
# ---------------------------------------------------------------------------

# Per short-time column, per channel: Hann taper then a length-1024 real FFT.
FLOPS_WINDOW = WINDOW  # 1024 multiplies for the Hann taper
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_POWER = 3 * N_BINS  # 1539 real/imag square-and-add for one auto-spectrum
FLOPS_PER_COL_PER_CH = FLOPS_WINDOW + FLOPS_RFFT + FLOPS_POWER  # 53763

# Per short-time column, all channels: the three cross-spectra W * conj(X|Y|Z), one complex multiply
# (four real multiplies plus two adds) per bin per pair.
FLOPS_CROSS_PER_PAIR_PER_BIN = 6
FLOPS_CROSS_PER_COL = 3 * FLOPS_CROSS_PER_PAIR_PER_BIN * N_BINS  # 9234

# Per short-time column: accumulate the four auto-spectra and the three complex cross-spectra (ten real
# scalars) from the 513 bins into the 64 bands.
FLOPS_BANDAGG_REALS = 10  # SWW, SXX, SYY, SZZ, Re/Im of SWX, SWY, SWZ
FLOPS_BANDAGG_PER_COL = FLOPS_BANDAGG_REALS * N_BINS  # 5130

# Per frame: sum the ten band-aggregated scalars over the COLS_PER_FRAME columns of the frame.
FLOPS_COL_TO_FRAME = FLOPS_BANDAGG_REALS * (COLS_PER_FRAME - 1) * N_BANDS  # 2560

# Per frame, per band: three magnitude-squared coherences and one directness from the aggregates.
FLOPS_MSC_PER_PAIR = 6  # |S|^2 (2 mul + 1 add), denom (1 mul), + eps (1 add), div (1)
FLOPS_DIRECTNESS = 12  # intensity norm (3 mul + 2 add + sqrt), energy (3 add + 1 mul), + eps, div
FLOPS_PER_BAND_FINAL = 3 * FLOPS_MSC_PER_PAIR + FLOPS_DIRECTNESS  # 30
FLOPS_BANDS = FLOPS_PER_BAND_FINAL * N_BANDS  # 1920

FLOPS_PER_FRAME = (
    FLOPS_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME
    + FLOPS_CROSS_PER_COL * COLS_PER_FRAME
    + FLOPS_BANDAGG_PER_COL * COLS_PER_FRAME
    + FLOPS_COL_TO_FRAME
    + FLOPS_BANDS
)  # 1_151_560


def _band_partition(sample_rate: int) -> np.ndarray:

    bin_freqs = np.linspace(0.0, sample_rate / 2.0, N_BINS)
    mel = hz_to_mel(bin_freqs)
    mel_max = hz_to_mel(np.array([sample_rate / 2.0]))[0]
    band_idx = np.floor(mel / mel_max * N_BANDS).astype(np.int64)
    band_idx = np.clip(band_idx, 0, N_BANDS - 1)
    matrix = np.zeros((N_BANDS, N_BINS), dtype=np.float64)
    matrix[band_idx, np.arange(N_BINS)] = 1.0
    return matrix


@dataclass(frozen=True, slots=True)
class InterchannelCoherenceFeaturizer(FrozenFeatureProvider):

    _flops_per_frame = FLOPS_PER_FRAME
    sample_rate: int = SAMPLE_RATE_HZ

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def band_partition(self) -> np.ndarray:
        return _band_partition(self.sample_rate)

    def parameter_digest(self) -> str:

        payload = {
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "band_partition_sha256": hashlib.sha256(
                self.band_partition.astype("<f8").tobytes()
            ).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "n_bands": N_BANDS,
            "n_spatial": N_SPATIAL,
            "sample_rate": self.sample_rate,
            "family": "interchannel_coherence",
        }
        return canonical_sha256(payload)

    def _channel_spectra(self, audio: np.ndarray, n_frames: int, n_cols: int) -> np.ndarray:

        window = self.window
        spectra = np.empty((N_CHANNELS, n_cols, N_BINS), dtype=np.complex128)
        for ch in range(N_CHANNELS):
            padded = np.zeros(n_frames * SAMPLES_PER_FRAME + PAD_RIGHT, dtype=np.float64)
            padded[: audio.shape[1]] = audio[ch]
            columns = np.empty((n_cols, WINDOW), dtype=np.float64)
            for c in range(n_cols):
                start = c * HOP
                columns[c] = padded[start : start + WINDOW] * window
            spectra[ch] = np.fft.rfft(columns, n=N_FFT, axis=1)
        return spectra

    def featurize(self, audio: np.ndarray) -> np.ndarray:

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise ValueError(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise ValueError("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        n_cols = n_frames * COLS_PER_FRAME

        spectra = self._channel_spectra(audio, n_frames, n_cols)
        w, x, y, z = spectra[0], spectra[1], spectra[2], spectra[3]

        # Per-bin auto power and the three W-vs-gradient cross-spectra.
        pww = (w.real * w.real) + (w.imag * w.imag)
        pxx = (x.real * x.real) + (x.imag * x.imag)
        pyy = (y.real * y.real) + (y.imag * y.imag)
        pzz = (z.real * z.real) + (z.imag * z.imag)
        cwx = w * np.conj(x)
        cwy = w * np.conj(y)
        cwz = w * np.conj(z)

        partition = self.band_partition
        partition_c = partition.astype(np.complex128)

        def _frame_agg_real(power: np.ndarray) -> np.ndarray:
            banded = power @ partition.T  # (n_cols, N_BANDS)
            return banded.reshape(n_frames, COLS_PER_FRAME, N_BANDS).sum(axis=1)

        def _frame_agg_complex(cross: np.ndarray) -> np.ndarray:
            banded = cross @ partition_c.T  # (n_cols, N_BANDS) complex
            return banded.reshape(n_frames, COLS_PER_FRAME, N_BANDS).sum(axis=1)

        s_ww = _frame_agg_real(pww)
        s_xx = _frame_agg_real(pxx)
        s_yy = _frame_agg_real(pyy)
        s_zz = _frame_agg_real(pzz)
        s_wx = _frame_agg_complex(cwx)
        s_wy = _frame_agg_complex(cwy)
        s_wz = _frame_agg_complex(cwz)

        # Magnitude-squared coherence per band, in [0, 1].
        msc_wx = ((s_wx.real ** 2) + (s_wx.imag ** 2)) / (s_ww * s_xx + _EPS)
        msc_wy = ((s_wy.real ** 2) + (s_wy.imag ** 2)) / (s_ww * s_yy + _EPS)
        msc_wz = ((s_wz.real ** 2) + (s_wz.imag ** 2)) / (s_ww * s_zz + _EPS)

        # DirAC directness = ||active intensity|| / energy, in [0, 1]. One minus diffuseness.
        i_x, i_y, i_z = s_wx.real, s_wy.real, s_wz.real
        intensity_norm = np.sqrt((i_x * i_x) + (i_y * i_y) + (i_z * i_z))
        energy = 0.5 * (s_ww + s_xx + s_yy + s_zz)
        directness = intensity_norm / (energy + _EPS)

        return np.concatenate([msc_wx, msc_wy, msc_wz, directness], axis=1)
