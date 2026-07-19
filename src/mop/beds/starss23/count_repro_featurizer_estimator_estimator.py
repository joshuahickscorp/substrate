
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import ZeroParameterProvider
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_REPRO_FE_ESTIMATOR_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-estimator/v1"

BETA = 0.90  # proportion-of-variance the leading eigenvalues must reach to be counted as a source direction
NOISE_FLOOR = 1e-6  # covariance-power silence floor; below it the estimate is 0, matching the sealed floor

MAX_ESTIMABLE_SOURCES = 4

ESTIMATOR_RULE = (
    "cumulative-energy source count on the 4x4 FOA spatial covariance: E=0 below NOISE_FLOOR power, else "
    "clamp(smallest k with sum(top-k eigenvalues)/trace >= BETA, 1, 4)"
)

FLOPS_COVARIANCE = 2 * SAMPLES_PER_FRAME * N_CHANNELS * N_CHANNELS  # 76,800 for a 4x2400 block
FLOPS_EIGEN_4X4 = 3_200  # fixed 4x4 symmetric eigenvalue solve budget
FLOPS_CUMULATIVE = 3 * MAX_ESTIMABLE_SOURCES  # 12 running-sum, normalize, and compare across up to 4 eigs
FLOPS_PER_REESTIMATE = FLOPS_COVARIANCE + FLOPS_EIGEN_4X4 + FLOPS_CUMULATIVE  # 80,012


class CountReproEstimatorRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReproCountEstimator(ZeroParameterProvider):

    beta: float = BETA
    noise_floor: float = NOISE_FLOOR

    def parameter_digest(self) -> str:

        payload = {
            "schema": COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
            "beta": float(self.beta),
            "noise_floor": float(self.noise_floor),
            "max_estimable_sources": MAX_ESTIMABLE_SOURCES,
            "rule": ESTIMATOR_RULE,
            "n_channels": N_CHANNELS,
            "samples_per_frame": SAMPLES_PER_FRAME,
        }
        return canonical_sha256(payload)

    def estimate_track(self, audio: np.ndarray) -> np.ndarray:

        array = np.asarray(audio, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != N_CHANNELS:
            raise CountReproEstimatorRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = array.shape[1]
        if n_samples == 0 or n_samples % SAMPLES_PER_FRAME != 0:
            raise CountReproEstimatorRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        beta = float(self.beta)
        track = np.empty(n_frames, dtype=np.int64)
        for t in range(n_frames):
            block = array[:, t * SAMPLES_PER_FRAME : (t + 1) * SAMPLES_PER_FRAME]
            cov = (block @ block.T) / float(SAMPLES_PER_FRAME)
            power = float(np.trace(cov))
            if power < float(self.noise_floor):
                track[t] = 0
                continue
            eigvals = np.linalg.eigvalsh(cov)
            descending = np.clip(eigvals[::-1], 0.0, None)  # l1 >= l2 >= l3 >= l4 >= 0
            total = float(descending.sum())
            if total <= 0.0:
                track[t] = 0
                continue
            cumulative = np.cumsum(descending) / total
            k = int(np.searchsorted(cumulative, beta - 1e-12) + 1)
            track[t] = min(MAX_ESTIMABLE_SOURCES, max(1, k))
        return track

    def flops_for_reestimations(self, k: int) -> int:

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise CountReproEstimatorRefusal("k must be a nonnegative integer")
        return FLOPS_PER_REESTIMATE * k
