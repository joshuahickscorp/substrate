
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import FrozenFeatureProvider
from .featurizer import hann_window
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
)
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

DOA_FEATURIZER_SCHEMA = "mop-starss23-doa-featurizer/v1"

COLS_PER_FRAME = SAMPLES_PER_FRAME // HOP  # 2400 // 480 = 5 columns per 100 ms frame, shared with SpatialDoa

D_FEAT_DOA = N_BANDS * FEATURES_PER_BAND  # 256 = 64 bands x 4 (direction-cosine rate-of-change + diffuseness)


FLOPS_FLUX_DIFF = (COLS_PER_FRAME * D_FEAT_DOA * 2) + ((COLS_PER_FRAME - 1) * D_FEAT_DOA)  # 3_584

FLOPS_PER_FRAME = FLOPS_STFT + FLOPS_INTENSITY + FLOPS_REDUCE + FLOPS_FLUX_DIFF  # 1_127_761

_EPS = 1e-12


class DoaFeaturizerRefusal(ValueError):
    pass


def _channel_spectra(signal: np.ndarray, n_cols: int, window: np.ndarray) -> np.ndarray:

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

    per_band = np.stack([dir_x, dir_y, dir_z, diffuseness], axis=-1)  # (n_cols, N_BANDS, 4)
    return per_band.reshape(n_cols, N_BANDS * FEATURES_PER_BAND)


@dataclass(frozen=True, slots=True)
class DoaFeaturizer(FrozenFeatureProvider):

    _flops_per_frame = FLOPS_PER_FRAME
    _frame_count_refusal = DoaFeaturizerRefusal
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

    def featurize(self, audio: np.ndarray) -> np.ndarray:

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
