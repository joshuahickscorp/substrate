"""Bias-independent reproduction (axis: featurizer_estimator), part B: a re-authored frozen count estimator.

This is a NET-NEW, ADDITIVE module. It edits no sealed count_* or onset module and no existing proof. It
pairs with the re-authored gammatone front-end to adversarially test whether the counting bed's first
mechanics-ok signal survives when BOTH the front-end the gate reads AND the count estimator the gate decides
to spend are swapped for independently authored zero-parameter DSP rules. If the win was an artifact of the
sealed eigenvalue-threshold estimator's particular error signature, it dies here; if the gate genuinely
places its fixed re-estimation budget near real count changes, it survives against a different estimator too.

What differs from the sealed ``count_estimator.FrozenCountEstimator``
--------------------------------------------------------------------
The sealed estimator counts eigenvalues that clear a fixed fraction of the leading eigenvalue,
``clamp(#{eig >= ALPHA * max_eig}, 1, 4)`` with ``ALPHA = 0.15``. This module instead uses a
CUMULATIVE-ENERGY (proportion-of-variance) rule on the SAME 4x4 FOA spatial covariance: it returns the
smallest number of leading eigenvalues whose running sum reaches a fixed fraction ``BETA`` of the total
covariance energy (the trace),

    K_t     = (S_t @ S_t.T) / SAMPLES_PER_FRAME              # 4x4 real symmetric PSD spatial covariance
    power   = trace(K_t)                                     # total covariance energy = sum of eigenvalues
    lambda  = eigvalsh(K_t) sorted descending               # l1 >= l2 >= l3 >= l4 >= 0
    E'(t)   = 0                                              if power < NOISE_FLOOR    (silence)
    E'(t)   = clamp( smallest k with sum(l_1..l_k)/power >= BETA, 1, 4 )   otherwise

with a fixed, preregistered, corpus-independent ``BETA = 0.90``. This is a genuinely different frozen track:
the eigenvalue-threshold rule keys on the RATIO of each eigenvalue to the largest, while the cumulative-energy
rule keys on how many directions are needed to explain a fixed share of the total power, so the two disagree
frame by frame. Both cap at 4 (the FOA rank ceiling) and both are threshold-sensitive, so this estimator is
genuinely imperfect and is not the label: there is no leakage.

Like the sealed estimator it carries ZERO trained parameters (``n_params() == 0``): ``BETA`` and
``NOISE_FLOOR`` are hand-set DSP priors, not learned. It is computed once per clip; the referee indexes into
the resulting track at the arm's re-estimation frames, so compute is charged per re-estimation actually
spent. Its analytic per-re-estimation FLOPs are its own anchor.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import ZeroParameterProvider
from .schema import N_CHANNELS, SAMPLES_PER_FRAME

COUNT_REPRO_FE_ESTIMATOR_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-estimator/v1"

# Fixed, preregistered DSP priors. Hand-set, corpus-independent, never label-tuned.
BETA = 0.90  # proportion-of-variance the leading eigenvalues must reach to be counted as a source direction
NOISE_FLOOR = 1e-6  # covariance-power silence floor; below it the estimate is 0, matching the sealed floor

# The FOA first-order rank ceiling: four channels resolve at most four distinct directions.
MAX_ESTIMABLE_SOURCES = 4

ESTIMATOR_RULE = (
    "cumulative-energy source count on the 4x4 FOA spatial covariance: E=0 below NOISE_FLOOR power, else "
    "clamp(smallest k with sum(top-k eigenvalues)/trace >= BETA, 1, 4)"
)

# Analytic FLOPs per re-estimation, reproducible across hosts, matching the sealed estimator's covariance and
# eigen-solve anchors plus the small cumulative-sum-and-compare of the proportion-of-variance rule.
FLOPS_COVARIANCE = 2 * SAMPLES_PER_FRAME * N_CHANNELS * N_CHANNELS  # 76,800 for a 4x2400 block
FLOPS_EIGEN_4X4 = 3_200  # fixed 4x4 symmetric eigenvalue solve budget
FLOPS_CUMULATIVE = 3 * MAX_ESTIMABLE_SOURCES  # 12 running-sum, normalize, and compare across up to 4 eigs
FLOPS_PER_REESTIMATE = FLOPS_COVARIANCE + FLOPS_EIGEN_4X4 + FLOPS_CUMULATIVE  # 80,012


class CountReproEstimatorRefusal(ValueError):
    """Raised when the estimator input violates the frozen FOA acquisition contract."""


@dataclass(frozen=True, slots=True)
class ReproCountEstimator(ZeroParameterProvider):
    """The re-authored zero-trained-parameter cumulative-energy count estimator. Deterministic per host."""

    beta: float = BETA
    noise_floor: float = NOISE_FLOOR

    def parameter_digest(self) -> str:
        """Digest sealing the fixed constants and rule string so the estimator is provably frozen."""

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
        """Return the (n_frames,) int64 per-frame count estimate over a ``(N_CHANNELS, n_samples)`` block.

        Deterministic and byte-reproducible on a given host: identical input bytes yield an identical track.
        The audio length must be a whole number of 100 ms frames, matching the frozen grid.
        """

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
            # smallest k (1-indexed) whose leading-eigenvalue energy reaches BETA of the total
            k = int(np.searchsorted(cumulative, beta - 1e-12) + 1)
            track[t] = min(MAX_ESTIMABLE_SOURCES, max(1, k))
        return track

    def flops_for_reestimations(self, k: int) -> int:
        """Analytic FLOPs for spending ``k`` re-estimations: k x FLOPS_PER_REESTIMATE. Reproducible."""

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise CountReproEstimatorRefusal("k must be a nonnegative integer")
        return FLOPS_PER_REESTIMATE * k
