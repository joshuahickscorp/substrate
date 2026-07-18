"""Direction-of-arrival bed, component 3: the frozen zero-trained-parameter wideband DoA estimator.

This is the WHAT signal: the fixed, non-trained re-estimation mechanism a re-estimating arm actually gets
when it spends a look. It is deliberately a DIFFERENT, cheaper computation than the featurizer (mirrors
``count_estimator.py`` being a different, cheaper computation than ``count_featurizer.py``): a single
WIDEBAND time-domain active-intensity direction per frame, not per-band, no FFT.

Per 100 ms frame, over its ``(N_CHANNELS, SAMPLES_PER_FRAME)`` block, ACN channel order ``(W, Y, Z, X)``
reused from ``featurizer_spatial_doa.ACN_W/Y/Z/X``:

    I_x = sum_n W(n) * X(n)     I_y = sum_n W(n) * Y(n)     I_z = sum_n W(n) * Z(n)
    energy = 0.5 * sum_n (W(n)^2 + X(n)^2 + Y(n)^2 + Z(n)^2)
    azimuth   = atan2(-I_y, -I_x)      (DirAC convention: direction opposite the intensity flow)
    elevation = atan2(-I_z, sqrt(I_x^2 + I_y^2))
    E(t) = (azimuth_deg, elevation_deg)          if energy >= NOISE_FLOOR
    E(t) = the referee's fixed cold-start boresight direction    otherwise (silence: numerically defined,
                                                                              never scored at that frame)

At about 33,687 FLOPs this is roughly 33x cheaper than the featurizer's 1,127,761 FLOPs per frame (so
re-estimation is not the dominant cost, coasting genuinely saves compute) and roughly 5x Architecture A's
6,385-FLOP gate inference (so choosing well about WHEN to spend it is a real, non-trivial saving), the
same qualitative relationship ``count_estimator.py`` establishes for counting.

The estimator track is computed ONCE per clip and shared across every arm, every seed, and BOTH gate
architectures. Only the re-estimation frame set differs per arm, per seed, and per architecture, mirroring
the counting bed's E-is-shared, R-varies design.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .doa_referee import DOA_COLD_START_AZIMUTH_DEG, DOA_COLD_START_ELEVATION_DEG
from .featurizer_spatial_doa import ACN_W, ACN_X, ACN_Y, ACN_Z
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

DOA_ESTIMATOR_SCHEMA = "mop-starss23-doa-estimator/v1"

# Reused constant value from count_estimator.py: same DSP-prior status, a fixed noise-power floor below
# which a frame is treated as silent and the estimator emits the cold-start direction.
NOISE_FLOOR = 1e-6

ESTIMATOR_RULE = (
    "wideband time-domain active-intensity direction: azimuth = atan2(-Iy, -Ix), "
    "elevation = atan2(-Iz, sqrt(Ix^2+Iy^2)); the cold-start boresight direction below NOISE_FLOOR power"
)

# Analytic FLOPs per re-estimation, reproducible across hosts.
FLOPS_INTENSITY_TD = 3 * 2 * SAMPLES_PER_FRAME  # 3 components x 2 (multiply, accumulate) x 2400 = 14_400
FLOPS_ENERGY_TD = 4 * 2 * SAMPLES_PER_FRAME  # 4 channels x 2 (square, accumulate) x 2400 = 19_200
FLOPS_COMBINE = 4  # 3 adds to combine channel energies + 1 scale by 0.5
FLOPS_REDUCE_ONCE = 83  # reused FLOPS_REDUCE_PER_BAND value from featurizer_spatial_doa.py, once per frame
FLOPS_PER_REESTIMATE = FLOPS_INTENSITY_TD + FLOPS_ENERGY_TD + FLOPS_COMBINE + FLOPS_REDUCE_ONCE  # 33_687


class DoaEstimatorRefusal(ValueError):
    """Raised when the estimator input violates the frozen FOA acquisition contract."""


@dataclass(frozen=True, slots=True)
class FrozenDoaEstimator:
    """The frozen, zero-trained-parameter wideband DoA estimator. Deterministic per host."""

    noise_floor: float = NOISE_FLOOR

    def n_params(self) -> int:
        """Zero trained parameters. The estimator is a fixed DSP rule, never a learned map."""

        return 0

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
        """Return the (n_frames, 2) float64 per-frame wideband direction estimate, in degrees.

        Deterministic and byte-reproducible on a given host: identical input bytes yield an identical
        track. The audio length must be a whole number of 100 ms frames, matching the frozen grid.
        """

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
        """Analytic FLOPs for spending ``k`` re-estimations: k x FLOPS_PER_REESTIMATE. Reproducible."""

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise DoaEstimatorRefusal("k must be a nonnegative integer")
        return FLOPS_PER_REESTIMATE * k
