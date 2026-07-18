"""Direction-of-arrival bed, component 2: the frozen zero-trained-parameter spatial-flux front-end.

This is the WHEN signal: "is the spatial field changing right now," not "what direction is it." It is a
net-new, additive front-end that reuses the DSP grid and per-band active-intensity math already built for
the onset bed's spatial-DOA front-end swap (``featurizer_spatial_doa.py``, itself additive and not
sealed), by import: ``WINDOW``, ``N_FFT``, ``HOP``, ``COLS_PER_FRAME``, ``PAD_RIGHT``, ``N_BANDS``,
``FEATURES_PER_BAND``, ``ACN_W/Y/Z/X``, and the private helpers ``_hann_window``, ``_band_edges``,
``_band_membership`` (this codebase already treats leading-underscore names as intra-package reusable,
e.g. ``count_gate.py`` imports ``gate._sigmoid``).

At each of the ``COLS_PER_FRAME = 5`` STFT columns this computes the SAME per-band direction and
diffuseness reduction ``SpatialDoaFeaturizer`` computes (64 bands times [dir_x, dir_y, dir_z,
diffuseness]), but at COLUMN resolution instead of pooled to one value per label frame. It then takes the
elementwise absolute difference between consecutive columns (full-wave, not half-wave rectified: a
direction reversing is exactly as much "change" as a direction advancing) and sums the five within-frame
differences into one 256-dim frame vector, mirroring the sealed featurizer's "aggregate the
``COLS_PER_FRAME`` columns of each label frame by summation" convention exactly. The flux at the very
first column of the whole clip is defined as zero (there is no prior column to differ against).

Output layout is PER-BAND INTERLEAVED, not block-major: feature index ``4*b + 0..3`` holds
``[dir_x, dir_y, dir_z, diffuseness]`` for band ``b``, so the 64 diffuseness-flux components sit at
indices ``3, 7, 11, ..., 255``. This is the layout ``doa_gate.py``'s online-state diffuseness-flux EMA
slices, mirroring ``count_gate.py``'s ``_POS_BLOCK`` slicing convention.

Zero trained parameters: ``n_params()`` returns 0 and ``parameter_digest()`` hashes only deterministic
functions of the sample rate and the transform grid, never a learned weight. Compute is charged
analytically, never measured, so the FLOP ledger is host-independent.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .featurizer_spatial_doa import (
    ACN_W,
    ACN_X,
    ACN_Y,
    ACN_Z,
    FEATURES_PER_BAND,
    FLOPS_INTENSITY,
    FLOPS_REDUCE,
    FLOPS_STFT,
    HOP,
    N_BANDS,
    N_FFT,
    PAD_RIGHT,
    WINDOW,
    _band_membership,
    _hann_window,
)
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

DOA_FEATURIZER_SCHEMA = "mop-starss23-doa-featurizer/v1"

COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame, shared with SpatialDoa

D_FEAT_DOA = N_BANDS * FEATURES_PER_BAND  # 256 = 64 bands x 4 (rate-of-change of 3 direction cosines + 1 diffuseness)

# Reduction FLOPs: reused BY IMPORT from featurizer_spatial_doa.py's own per-frame reduction budget, which
# already charges the direction/diffuseness math per band per frame; charging it identically here whether
# the reduction happens at frame or column resolution keeps the two front-ends' per-unit cost comparable.

# New: the flux difference (5 columns x 256 dims x 2 FLOPs [subtract, abs] = 2560), plus summing the 5
# within-frame column differences into 1 per dim (4 adds x 256 = 1024).
FLOPS_FLUX_DIFF = (COLS_PER_FRAME * D_FEAT_DOA * 2) + ((COLS_PER_FRAME - 1) * D_FEAT_DOA)  # 3_584

FLOPS_PER_FRAME = FLOPS_STFT + FLOPS_INTENSITY + FLOPS_REDUCE + FLOPS_FLUX_DIFF  # 1_127_761

_EPS = 1e-12


class DoaFeaturizerRefusal(ValueError):
    """Raised when the DoA featurizer input violates the frozen FOA acquisition contract."""


def _channel_spectra(signal: np.ndarray, n_cols: int, window: np.ndarray) -> np.ndarray:
    """Windowed rFFT of one channel at COLUMN resolution: (n_cols, N_BINS) complex128. Deterministic.

    Structurally identical to ``featurizer_spatial_doa.SpatialDoaFeaturizer._channel_spectra`` (same
    WINDOW/N_FFT/HOP/PAD_RIGHT grid), reimplemented as a free function here because that method is bound
    to an instance's cached window property and is not part of this module's declared reuse-by-import
    surface.
    """

    padded = np.zeros(n_cols * HOP + PAD_RIGHT, dtype=np.float64)
    padded[: signal.shape[0]] = signal
    columns = np.empty((n_cols, WINDOW), dtype=np.float64)
    for c in range(n_cols):
        start = c * HOP
        columns[c] = padded[start : start + WINDOW]
    columns *= window
    return np.fft.rfft(columns, n=N_FFT, axis=1)


def _per_column_direction_features(
    audio: np.ndarray, band_membership: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """Per-STFT-column direction cosines and diffuseness, per band, interleaved: (n_cols, D_FEAT_DOA).

    Same active-intensity reduction ``SpatialDoaFeaturizer.featurize`` performs (active-intensity vector
    ``I = Re{conj(W) [X, Y, Z]}``, DirAC energy density, azimuth/elevation reduced to direction cosines,
    diffuseness as one minus the intensity-to-energy ratio), but evaluated at COLUMN resolution: the
    per-band pooling over the five columns of a label frame is skipped here so a rate-of-change over
    columns can be measured afterward.
    """

    n_samples = audio.shape[1]
    n_frames = n_samples // SAMPLES_PER_FRAME
    n_cols = n_frames * COLS_PER_FRAME
    spectra = [_channel_spectra(audio[ch], n_cols, window) for ch in range(N_CHANNELS)]
    w, x, y, z = spectra[ACN_W], spectra[ACN_X], spectra[ACN_Y], spectra[ACN_Z]

    ix = w.real * x.real + w.imag * x.imag
    iy = w.real * y.real + w.imag * y.imag
    iz = w.real * z.real + w.imag * z.imag
    energy = 0.5 * (
        (w.real * w.real + w.imag * w.imag)
        + (x.real * x.real + x.imag * x.imag)
        + (y.real * y.real + y.imag * y.imag)
        + (z.real * z.real + z.imag * z.imag)
    )

    sum_ix = ix @ band_membership.T  # (n_cols, N_BANDS)
    sum_iy = iy @ band_membership.T
    sum_iz = iz @ band_membership.T
    sum_e = energy @ band_membership.T

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

    # Per-band interleave: [dx_0, dy_0, dz_0, dif_0, dx_1, dy_1, dz_1, dif_1, ...]. Diffuseness then sits
    # at indices 3, 7, 11, ..., 255.
    per_band = np.stack([dir_x, dir_y, dir_z, diffuseness], axis=-1)  # (n_cols, N_BANDS, 4)
    return per_band.reshape(n_cols, N_BANDS * FEATURES_PER_BAND)


@dataclass(frozen=True, slots=True)
class DoaFeaturizer:
    """The frozen zero-trained-parameter spatial-flux front-end. Deep-frozen: window and band grid fixed."""

    sample_rate: int = 24_000

    @property
    def window(self) -> np.ndarray:
        return _hann_window()

    @property
    def band_membership(self) -> np.ndarray:
        return _band_membership(self.sample_rate)

    def n_params(self) -> int:
        """Zero trained parameters. The front-end is a deterministic spatial DSP, never a learned map."""

        return 0

    def parameter_digest(self) -> str:
        """Digest of the fixed window and band grid bytes, proving the front-end is byte-frozen."""

        membership = self.band_membership
        payload = {
            "schema": DOA_FEATURIZER_SCHEMA,
            "front_end": "doa_spatial_flux_of_change",
            "window_sha256": hashlib.sha256(self.window.astype("<f8").tobytes()).hexdigest(),
            "band_membership_sha256": hashlib.sha256(membership.astype("<f8").tobytes()).hexdigest(),
            "window": WINDOW,
            "n_fft": N_FFT,
            "hop": HOP,
            "cols_per_frame": COLS_PER_FRAME,
            "n_bands": N_BANDS,
            "features_per_band": FEATURES_PER_BAND,
            "d_feat_doa": D_FEAT_DOA,
            "acn_channel_map": {"W": ACN_W, "Y": ACN_Y, "Z": ACN_Z, "X": ACN_X},
            "sample_rate": self.sample_rate,
        }
        return canonical_sha256(payload)

    def flops_for_frames(self, n_frames: int) -> int:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise DoaFeaturizerRefusal("n_frames must be a nonnegative integer")
        return FLOPS_PER_FRAME * n_frames

    def featurize(self, audio: np.ndarray) -> np.ndarray:
        """Featurize a (N_CHANNELS, n_samples) FOA array into (n_frames, D_FEAT_DOA=256) float64.

        Byte-reproducible: identical input bytes yield identical output bytes on a given host.
        """

        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 2 or audio.shape[0] != N_CHANNELS:
            raise DoaFeaturizerRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = audio.shape[1]
        if n_samples % SAMPLES_PER_FRAME != 0:
            raise DoaFeaturizerRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        if n_frames == 0:
            return np.zeros((0, D_FEAT_DOA), dtype=np.float64)

        col_features = _per_column_direction_features(audio, self.band_membership, self.window)
        flux = np.zeros_like(col_features)
        flux[1:] = np.abs(col_features[1:] - col_features[:-1])  # flux[0] = 0: no prior column in the clip
        frame_flux = flux.reshape(n_frames, COLS_PER_FRAME, D_FEAT_DOA).sum(axis=1)
        return np.ascontiguousarray(frame_flux, dtype=np.float64)

    def feature_digest(self, features: np.ndarray) -> str:
        """Digest of a feature block, used to assert byte-reproducibility of the front-end."""

        return hashlib.sha256(np.ascontiguousarray(features, dtype="<f8").tobytes()).hexdigest()
