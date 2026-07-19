
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import ZeroParameterProvider
from .doa_referee import DOA_COLD_START_AZIMUTH_DEG, DOA_COLD_START_ELEVATION_DEG
from .featurizer_spatial_doa import ACN_W, ACN_X, ACN_Y, ACN_Z
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

DOA_ESTIMATOR_SCHEMA = "mop-starss23-doa-estimator/v1"

NOISE_FLOOR = 1e-6

ESTIMATOR_RULE = (
    "wideband time-domain active-intensity direction: azimuth = atan2(-Iy, -Ix), "
    "elevation = atan2(-Iz, sqrt(Ix^2+Iy^2)); the cold-start boresight direction below NOISE_FLOOR power"
)

FLOPS_INTENSITY_TD = 3 * 2 * SAMPLES_PER_FRAME  # 3 components x 2 (multiply, accumulate) x 2400 = 14_400
FLOPS_ENERGY_TD = 4 * 2 * SAMPLES_PER_FRAME  # 4 channels x 2 (square, accumulate) x 2400 = 19_200
FLOPS_COMBINE = 4  # 3 adds to combine channel energies + 1 scale by 0.5
FLOPS_REDUCE_ONCE = 83  # reused FLOPS_REDUCE_PER_BAND value from featurizer_spatial_doa.py, once per frame
FLOPS_PER_REESTIMATE = FLOPS_INTENSITY_TD + FLOPS_ENERGY_TD + FLOPS_COMBINE + FLOPS_REDUCE_ONCE  # 33_687


class DoaEstimatorRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenDoaEstimator(ZeroParameterProvider):

    noise_floor: float = NOISE_FLOOR

    def parameter_digest(self) -> str:
        payload = {
            "schema": DOA_ESTIMATOR_SCHEMA,
            "noise_floor": float(self.noise_floor),
            "rule": ESTIMATOR_RULE,
            "cold_start_azimuth_deg": DOA_COLD_START_AZIMUTH_DEG,
            "cold_start_elevation_deg": DOA_COLD_START_ELEVATION_DEG,
            "n_channels": N_CHANNELS,
            "samples_per_frame": SAMPLES_PER_FRAME,
        }
        return canonical_sha256(payload)

    def estimate_track(self, audio: np.ndarray) -> np.ndarray:

        array = np.asarray(audio, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != N_CHANNELS:
            raise DoaEstimatorRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = array.shape[1]
        if n_samples == 0 or n_samples % SAMPLES_PER_FRAME != 0:
            raise DoaEstimatorRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME

        w = array[ACN_W].reshape(n_frames, SAMPLES_PER_FRAME)
        x = array[ACN_X].reshape(n_frames, SAMPLES_PER_FRAME)
        y = array[ACN_Y].reshape(n_frames, SAMPLES_PER_FRAME)
        z = array[ACN_Z].reshape(n_frames, SAMPLES_PER_FRAME)

        i_x = (w * x).sum(axis=1)
        i_y = (w * y).sum(axis=1)
        i_z = (w * z).sum(axis=1)
        energy = 0.5 * ((w * w).sum(axis=1) + (x * x).sum(axis=1) + (y * y).sum(axis=1) + (z * z).sum(axis=1))

        azimuth_deg = np.degrees(np.arctan2(-i_y, -i_x))
        elevation_deg = np.degrees(np.arctan2(-i_z, np.sqrt(i_x * i_x + i_y * i_y)))

        silent = energy < float(self.noise_floor)
        azimuth_deg = np.where(silent, DOA_COLD_START_AZIMUTH_DEG, azimuth_deg)
        elevation_deg = np.where(silent, DOA_COLD_START_ELEVATION_DEG, elevation_deg)

        return np.ascontiguousarray(np.stack([azimuth_deg, elevation_deg], axis=1), dtype=np.float64)

    def flops_for_reestimations(self, k: int) -> int:

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise DoaEstimatorRefusal("k must be a nonnegative integer")
        return FLOPS_PER_REESTIMATE * k
