"""A net-new FROZEN zero-trained-parameter spatial direction-of-arrival featurizer.

This is an additive front-end swap for the STARSS23 ESCS event-formation bed. It changes NO sealed
scoring module. Where the committed ``FrozenFeaturizer`` emits half-wave-rectified log-mel spectral flux,
this featurizer emits per-band ACTIVE-INTENSITY DIRECTION-OF-ARRIVAL features computed from the FOA
B-format: per frequency band the active-intensity vector I = Re{conj(W) [X, Y, Z]} is reduced to a
source azimuth and elevation (encoded as wraparound-free direction cosines) plus a DirAC diffuseness. It
carries the identical interface as the frozen featurizer (``featurize``, ``n_params``,
``parameter_digest``, ``feature_digest``, ``flops_for_frames`` and a module-level ``FLOPS_PER_FRAME``),
so it feeds the SAME unchanged trained gate through the SAME sealed harness, referee, and controls.

Physical motivation (docs/ESCS_DEEP_RESEARCH.md spatial cue). A new sound event onset introduces energy
arriving from a new direction. Plain log-mel energy discards the spatial channel entirely: two events at
the same loudness from different directions look identical to it. The active-intensity DOA makes that
spatial structure explicit, and the diffuseness flags frames where the field is incoherent (reverberant
tails, silence, isotropic noise) so the estimate is untrustworthy. Whether the trained gate can USE that
spatial structure to place a fixed firing budget better than uniform-random placement is exactly the
falsifiable question the bed scores; nothing here presupposes it can.

Zero trained parameters, asserted the same way the base featurizer asserts it: ``n_params()`` returns 0
and ``parameter_digest()`` hashes only deterministic functions of the sample rate and the transform grid
(the Hann window and the fixed triangular-free 0/1 band-membership matrix), never a learned weight. The
pipeline is float64 throughout and byte-reproducible: identical input bytes yield identical output bytes.

Output dimension. The featurizer emits exactly ``D_FEAT = 256`` per-frame features (64 frequency bands
times 4 spatial features: three direction cosines and one diffuseness), so the unchanged gate, whose
input contract is hardcoded to a length-256 feature vector, consumes it with NO projection and NO
truncation. This is deliberate: matching 256 natively keeps the gate, its 3193-parameter count, the
4096 ceiling, and the whole matched-budget story byte-identical to every other arm.

Compute is charged analytically, never measured, so the FLOP ledger is host-independent. The per-frame
cost is dominated by the same four-channel short-time transform the frozen front-end runs, so the DOA
front-end costs almost exactly what the frozen front-end costs and the matched-budget ceiling story is
unchanged.

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

# The short-time transform grid. Deliberately identical to the frozen featurizer so the audio-length
# contract (a whole number of 100 ms frames) and the exact five-columns-per-frame tiling are shared and
# the per-frame transform cost is comparable. The front-end owns these; the schema owns the label grid.
WINDOW = 1024
N_FFT = 1024
N_BINS = N_FFT // 2 + 1  # 513 one-sided rFFT bins
HOP = 480
COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame
PAD_RIGHT = WINDOW - HOP  # 544; makes exactly COLS_PER_FRAME * n_frames full-window columns

# Spatial-feature geometry. Four spatial scalars per band: three source-direction cosines (a wraparound
# free encoding of the azimuth and elevation) and one DirAC diffuseness. 64 bands times 4 equals 256.
N_BANDS = 64
FEATURES_PER_BAND = 4
D_FEAT = N_BANDS * FEATURES_PER_BAND  # 256 per-frame features; matches the gate's hardcoded contract

# FOA B-format channel map. STARSS23 FOA is delivered in ACN channel ordering with SN3D normalization,
# so the four channels are (W, Y, Z, X) = (omni, dipole-y, dipole-z, dipole-x). The active-intensity
# vector components pair the omni against each dipole. This ordering is a fixed corpus constant, not a
# knob; it is recorded in the parameter digest so a mis-map cannot masquerade as the same front-end.
ACN_W, ACN_Y, ACN_Z, ACN_X = 0, 1, 2, 3

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Analytic per-frame FLOP budget. Every term is a fixed integer so the ledger is reproducible across
# hosts. The dominant cost is the same four-channel windowed rFFT the frozen front-end runs.
# ---------------------------------------------------------------------------

# (a) The short-time transform, charged per column per channel exactly as the frozen ledger does.
FLOPS_WINDOW = WINDOW  # 1024 multiplies for the Hann taper
FLOPS_RFFT = 5 * N_FFT * 10  # 5 * 1024 * log2(1024) = 51200
FLOPS_STFT_PER_COL_PER_CH = FLOPS_WINDOW + FLOPS_RFFT  # 52224
FLOPS_STFT = FLOPS_STFT_PER_COL_PER_CH * N_CHANNELS * COLS_PER_FRAME  # 1_044_480

# (b) The active-intensity and energy-density arithmetic, charged per column per rFFT bin. Three
# intensity components Re{conj(W) [X,Y,Z]} at three FLOPs each (two multiplies, one add) is 9; four
# squared magnitudes |W|^2..|Z|^2 at three FLOPs each is 12; forming the energy density
# 0.5*(|W|^2+|X|^2+|Y|^2+|Z|^2) is 4 (three adds, one scale); accumulating the four band sums is 4.
FLOPS_INTENSITY_PER_COL_PER_BIN = 9 + 12 + 4 + 4  # 29
FLOPS_INTENSITY = FLOPS_INTENSITY_PER_COL_PER_BIN * COLS_PER_FRAME * N_BINS  # 29 * 5 * 513 = 74_385

# (c) The per-band reduction to a direction and a diffuseness, charged per band per frame. Norm of the
# intensity vector (13), diffuseness one-minus-ratio (3), azimuth atan2 (10), horizontal norm (11),
# elevation atan2 (10), four trig evaluations for the direction cosines (32), two cosine products (2),
# and a clip of the diffuseness (2).
FLOPS_REDUCE_PER_BAND = 13 + 3 + 10 + 11 + 10 + 32 + 2 + 2  # 83
FLOPS_REDUCE = FLOPS_REDUCE_PER_BAND * N_BANDS  # 83 * 64 = 5_312

FLOPS_PER_FRAME = FLOPS_STFT + FLOPS_INTENSITY + FLOPS_REDUCE  # 1_044_480 + 74_385 + 5_312 = 1_124_177


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * float(np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (float(np.power(10.0, mel / 2595.0)) - 1.0)


def _band_edges(sample_rate: int) -> np.ndarray:
    """Return N_BANDS+1 strictly increasing integer rFFT-bin edges, mel-spaced, covering all N_BINS bins.

    The band partition is a deterministic function of the sample rate and the transform size, never a
    learned quantity. Each of the N_BANDS bands owns a contiguous, non-empty range of rFFT bins, and the
    bands tile [0, N_BINS) exactly. Mel spacing puts finer bands at low frequency; the strictly
    increasing clamp guarantees no band is empty even where the mel grid is narrower than one bin.
    """

    f_max = sample_rate / 2.0
    mel_lo, mel_hi = _hz_to_mel(0.0), _hz_to_mel(f_max)
    mel_points = np.linspace(mel_lo, mel_hi, N_BANDS + 1)
    hz_points = np.array([_mel_to_hz(float(m)) for m in mel_points], dtype=np.float64)
    raw = np.round(hz_points / f_max * (N_BINS - 1)).astype(np.int64)
    edges = np.empty(N_BANDS + 1, dtype=np.int64)
    edges[0] = 0
    edges[N_BANDS] = N_BINS
    for b in range(1, N_BANDS):
        # Keep the edge strictly increasing and leave room for the remaining bands to stay non-empty.
        lo = edges[b - 1] + 1
        hi = N_BINS - (N_BANDS - b)
        edges[b] = int(min(max(int(raw[b]), lo), hi))
    return edges


def _band_membership(sample_rate: int) -> np.ndarray:
    """Return the fixed (N_BANDS, N_BINS) 0/1 band-membership matrix. Zero trained parameters."""

    edges = _band_edges(sample_rate)
    membership = np.zeros((N_BANDS, N_BINS), dtype=np.float64)
    for b in range(N_BANDS):
        membership[b, edges[b] : edges[b + 1]] = 1.0
    return membership


@dataclass(frozen=True, slots=True)
class SpatialDoaFeaturizer(FrozenFeatureProvider):
    """The frozen active-intensity DOA front-end. Deep-frozen: window and band grid are fixed DSP.

    The featurizer holds no trainable state. Its only constants are the Hann window and the band
    membership matrix, both deterministic functions of the sample rate and the transform grid. It emits
    a length-256 per-frame vector so the unchanged gate consumes it directly.
    """

    _flops_per_frame = FLOPS_PER_FRAME
    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return hann_window()

    @property
    def band_membership(self) -> np.ndarray:
        return _band_membership(self.sample_rate)

    def parameter_digest(self) -> str:
        """Digest of the fixed window and band grid bytes, proving the front-end is byte-frozen."""

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
        """Return the windowed rFFT of one channel: (n_cols, N_BINS) complex128. Deterministic."""

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
        """Featurize a (N_CHANNELS, n_samples) FOA array into (n_frames, D_FEAT=256) float64.

        Byte-reproducible: identical input bytes yield identical output bytes on a given host. The output
        is laid out as four contiguous 64-band blocks: the three source-direction cosines (from the
        active-intensity azimuth and elevation) followed by the DirAC diffuseness.
        """

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

        # Active-intensity vector I = Re{conj(W) [X, Y, Z]} per column per bin, and the DirAC energy
        # density E' = 0.5 (|W|^2 + |X|^2 + |Y|^2 + |Z|^2). Both are real, deterministic per bin.
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
            # Average over the five columns of each label frame and over the bins of each band. The 1/N
            # normalization cancels in the diffuseness ratio and the direction, so plain sums suffice.
            per_frame = field.reshape(n_frames, COLS_PER_FRAME, N_BINS).sum(axis=1)
            return per_frame @ membership.T  # (n_frames, N_BANDS)

        sum_ix = _band_sum(ix)
        sum_iy = _band_sum(iy)
        sum_iz = _band_sum(iz)
        sum_e = _band_sum(energy)

        intensity_norm = np.sqrt(sum_ix * sum_ix + sum_iy * sum_iy + sum_iz * sum_iz)
        diffuseness = np.clip(1.0 - intensity_norm / (sum_e + _EPS), 0.0, 1.0)

        # Source direction is opposite the active-intensity flow (DirAC convention). Reduce it to an
        # azimuth and an elevation, then re-encode as wraparound-free direction cosines.
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
