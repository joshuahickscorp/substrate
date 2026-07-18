"""Concurrent-source-counting bed, component 3a: the frozen zero-parameter count estimator.

This is the re-estimation the gate decides whether to spend. It carries ZERO trained parameters: it is a
classic eigenvalue-threshold source-number estimator on the 4x4 FOA spatial covariance. More distinct
source directions raise the numerical rank of the covariance, so counting the eigenvalues that clear a
fixed fraction of the leading eigenvalue estimates the number of concurrent directions.

Per 100 ms frame t, over its ``N_CHANNELS x SAMPLES_PER_FRAME`` FOA block ``S_t``:

    K_t     = (S_t @ S_t.T) / SAMPLES_PER_FRAME              # 4x4 real symmetric PSD spatial covariance
    lambda  = eigvalsh(K_t) sorted descending               # l1 >= l2 >= l3 >= l4 >= 0
    power   = trace(K_t)
    E(t)    = 0                                              if power < NOISE_FLOOR    (silence)
    E(t)    = clamp( #{k : lambda_k >= ALPHA * l1}, 1, 4 )   otherwise

The constants ``ALPHA`` and ``NOISE_FLOOR`` are fixed, preregistered, corpus-independent DSP priors,
hand-set exactly like the mel filterbank constants of the frozen featurizer, NOT tuned on labels. The
estimator is genuinely imperfect: it caps at 4 (the FOA rank ceiling, so it cannot resolve the rare
C=5) and is threshold-sensitive, so it is not the label and there is no leakage. It is computed once per
clip; the referee indexes into the resulting track at the arm's re-estimation frames, so compute is
charged per re-estimation actually spent (K x FLOPS_PER_REESTIMATE), not per frame.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_ESTIMATOR_SCHEMA = "mop-starss23-count-estimator/v1"

# Fixed, preregistered DSP priors. Hand-set, corpus-independent, never label-tuned.
ALPHA = 0.15
NOISE_FLOOR = 1e-6

# The FOA first-order rank ceiling: four channels resolve at most four distinct directions.
MAX_ESTIMABLE_SOURCES = 4

ESTIMATOR_RULE = (
    "eigenvalue-threshold source count on the 4x4 FOA spatial covariance: E=0 below NOISE_FLOOR power, "
    "else clamp(#{eig >= ALPHA * max_eig}, 1, 4)"
)

# Analytic FLOPs per re-estimation, reproducible across hosts. The 4x4 spatial covariance S @ S.T over a
# 2400-sample block is 16 inner products of length 2400 (about 2 * 2400 * 16 = 76,800 multiply-add) and a
# 4x4 symmetric eigenvalue solve is a fixed few hundred to a thousand FLOPs. The sum is anchored at 80,000.
FLOPS_COVARIANCE = 2 * SAMPLES_PER_FRAME * N_CHANNELS * N_CHANNELS  # 76,800 for a 4x2400 block
FLOPS_EIGEN_4X4 = 3_200  # fixed 4x4 symmetric eigenvalue solve budget
FLOPS_PER_REESTIMATE = FLOPS_COVARIANCE + FLOPS_EIGEN_4X4  # 80,000


class CountEstimatorRefusal(ValueError):
    """Raised when the estimator input violates the frozen FOA acquisition contract."""


@dataclass(frozen=True, slots=True)
class FrozenCountEstimator:
    """The frozen, zero-trained-parameter concurrent-source count estimator. Deterministic per host."""

    alpha: float = ALPHA
    noise_floor: float = NOISE_FLOOR

    def n_params(self) -> int:
        """Zero trained parameters. The estimator is a fixed DSP rule, never a learned counter."""

        return 0

    def parameter_digest(self) -> str:
        """Digest sealing the fixed constants and rule string so the estimator is provably frozen."""

        payload = {
            "schema": COUNT_ESTIMATOR_SCHEMA,
            "alpha": float(self.alpha),
            "noise_floor": float(self.noise_floor),
            "max_estimable_sources": MAX_ESTIMABLE_SOURCES,
            "rule": ESTIMATOR_RULE,
            "n_channels": N_CHANNELS,
            "samples_per_frame": SAMPLES_PER_FRAME,
        }
        return canonical_sha256(payload)

    def estimate_track(self, audio: np.ndarray) -> np.ndarray:
        """Return the (n_frames,) int64 per-frame count estimate over a ``(N_CHANNELS, n_samples)`` block.

        Deterministic and byte-reproducible on a given host: identical input bytes yield an identical
        track. The audio length must be a whole number of 100 ms frames, matching the frozen grid.
        """

        array = np.asarray(audio, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != N_CHANNELS:
            raise CountEstimatorRefusal(f"audio must be shape ({N_CHANNELS}, n_samples)")
        n_samples = array.shape[1]
        if n_samples == 0 or n_samples % SAMPLES_PER_FRAME != 0:
            raise CountEstimatorRefusal("audio length must be a whole number of 100 ms frames")
        n_frames = n_samples // SAMPLES_PER_FRAME
        track = np.empty(n_frames, dtype=np.int64)
        for t in range(n_frames):
            block = array[:, t * SAMPLES_PER_FRAME : (t + 1) * SAMPLES_PER_FRAME]
            cov = (block @ block.T) / float(SAMPLES_PER_FRAME)
            power = float(np.trace(cov))
            if power < float(self.noise_floor):
                track[t] = 0
                continue
            eigvals = np.linalg.eigvalsh(cov)
            leading = float(eigvals[-1])
            if leading <= 0.0:
                track[t] = 0
                continue
            threshold = float(self.alpha) * leading
            count = int(np.count_nonzero(eigvals >= threshold))
            track[t] = min(MAX_ESTIMABLE_SOURCES, max(1, count))
        return track

    def flops_for_reestimations(self, k: int) -> int:
        """Analytic FLOPs for spending ``k`` re-estimations: k x FLOPS_PER_REESTIMATE. Reproducible."""

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise CountEstimatorRefusal("k must be a nonnegative integer")
        return FLOPS_PER_REESTIMATE * k
