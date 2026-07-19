
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.science.budget import ARM_ALWAYS_ON, ARM_BEST_SINGLE, ARM_RATE_MATCHED_RANDOM
from mop.substrate.events import canonical_sha256

from .adapter import domain_seed
from .featurizer import D_FEAT
from .referee import score_arm
from .schema import COLLAR_FRAMES

CONTROLS_SCHEMA = "mop-starss23-escs-controls/v1"

_RMR_NAMESPACE = "mop.beds.starss23.controls.rate_matched_random"
_NOISY_TV_NAMESPACE = "mop.beds.starss23.controls.noisy_tv"
_RND_TARGET_NAMESPACE = "mop.beds.starss23.controls.rnd_target"


DEFAULT_NOISY_TV_TOLERANCE = 0.05


class ControlRefusal(ValueError):
    pass


def _unique_sorted_frames(frames: Sequence[int], n_frames: int, label: str) -> list[int]:
    seen: set[int] = set()
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise ControlRefusal(f"{label} must contain frames in [0, {n_frames})")
        seen.add(frame)
    return sorted(seen)




def rate_matched_random_fires(
    candidate_fires: Sequence[int], n_frames: int, *, seed: int, clip_id: str
) -> list[int]:

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ControlRefusal("clip_id must be a non-empty string")
    fires = _unique_sorted_frames(candidate_fires, n_frames, "candidate_fires")
    k = len(fires)
    if k >= n_frames:
        return list(range(n_frames))
    rng = np.random.default_rng(
        domain_seed(seed, f"{_RMR_NAMESPACE}:{clip_id}", b"mop-starss23-controls-v1")
    )
    positions = rng.choice(n_frames, size=k, replace=False)
    return sorted(int(position) for position in positions)


@dataclass(frozen=True, slots=True)
class RateMatchedRandomControl:

    seed: int
    arm_kind: str = ARM_RATE_MATCHED_RANDOM

    def fires_for_clip(self, candidate_fires: Sequence[int], n_frames: int, clip_id: str) -> list[int]:
        return rate_matched_random_fires(
            candidate_fires, n_frames, seed=self.seed, clip_id=clip_id
        )




def always_on_fires(n_frames: int) -> list[int]:

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return list(range(n_frames))


def never_update_reestimates(n_frames: int) -> list[int]:

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return []


def frame_flux(features: np.ndarray) -> np.ndarray:

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != D_FEAT:
        raise ControlRefusal(f"features must be shape (n_frames, {D_FEAT})")
    return array.sum(axis=1)


def best_single_fires(flux: np.ndarray, threshold: float) -> list[int]:

    array = np.asarray(flux, dtype=np.float64)
    if array.ndim != 1:
        raise ControlRefusal("flux must be a 1-D per-frame array")
    return [int(index) for index in np.nonzero(array >= float(threshold))[0]]


def _candidate_thresholds(flux_values: np.ndarray) -> list[float]:

    unique = np.unique(flux_values)
    if unique.size == 0:
        return [0.0]
    midpoints = ((unique[:-1] + unique[1:]) / 2.0).tolist() if unique.size > 1 else []
    return [float(unique[0] - 1.0), *[float(value) for value in midpoints], float(unique[-1] + 1.0)]


def tune_best_single_threshold(
    val_clips: Sequence[tuple[np.ndarray, Sequence[int]]], *, collar: int = COLLAR_FRAMES
) -> float:

    if not val_clips:
        raise ControlRefusal("best-single tuning needs at least one val clip")
    per_clip_flux = [frame_flux(features) for features, _ in val_clips]
    pooled = np.concatenate(per_clip_flux) if per_clip_flux else np.zeros(0)
    thresholds = _candidate_thresholds(pooled)
    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        scored = [
            (list(gt_frames), best_single_fires(flux, threshold))
            for flux, (_, gt_frames) in zip(per_clip_flux, val_clips, strict=True)
        ]
        f1 = score_arm(scored, collar).f1
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


@dataclass(frozen=True, slots=True)
class BestSingleControl:

    threshold: float
    arm_kind: str = ARM_BEST_SINGLE

    @classmethod
    def tuned(
        cls, val_clips: Sequence[tuple[np.ndarray, Sequence[int]]], *, collar: int = COLLAR_FRAMES
    ) -> BestSingleControl:
        return cls(threshold=tune_best_single_threshold(val_clips, collar=collar))

    def fires_for_clip(self, features: np.ndarray) -> list[int]:
        return best_single_fires(frame_flux(features), self.threshold)


@dataclass(frozen=True, slots=True)
class AlwaysOnControl:

    arm_kind: str = ARM_ALWAYS_ON

    def fires_for_clip(self, n_frames: int) -> list[int]:
        return always_on_fires(n_frames)




@dataclass(frozen=True, slots=True)
class RndTarget:

    seed: int
    d_in: int = D_FEAT
    d_out: int = 32

    def _matrix(self) -> np.ndarray:
        rng = np.random.default_rng(
            domain_seed(self.seed, _RND_TARGET_NAMESPACE, b"mop-starss23-controls-v1")
        )
        return rng.standard_normal((self.d_out, self.d_in)) / np.sqrt(self.d_in)

    def novelty(self, features: np.ndarray) -> np.ndarray:

        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.d_in:
            raise ControlRefusal(f"features must be shape (n_frames, {self.d_in})")
        projected = array @ self._matrix().T
        return np.sqrt((projected * projected).sum(axis=1))


def pure_aleatoric_channel(seed: int, n_frames: int, *, d_feat: int = D_FEAT) -> np.ndarray:

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    rng = np.random.default_rng(
        domain_seed(seed, _NOISY_TV_NAMESPACE, b"mop-starss23-controls-v1")
    )
    return rng.standard_normal((n_frames, d_feat))


def noise_chasing_fire_fn(
    normal_features: np.ndarray, *, base_rate: float, seed: int
) -> Callable[[np.ndarray], bool]:

    if not 0.0 < float(base_rate) < 1.0:
        raise ControlRefusal("base_rate must be a probability strictly between 0 and 1")
    target = RndTarget(seed=seed)
    normal_novelty = target.novelty(normal_features)
    threshold = float(np.quantile(normal_novelty, 1.0 - float(base_rate)))

    def _fire(features_row: np.ndarray) -> bool:
        row = np.asarray(features_row, dtype=np.float64).reshape(1, -1)
        return bool(target.novelty(row)[0] >= threshold)

    return _fire


def firing_rate_on_frames(fire_fn: Callable[[np.ndarray], bool], features: np.ndarray) -> float:

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ControlRefusal("features must be a 2-D (n_frames, d_feat) block")
    if array.shape[0] == 0:
        raise ControlRefusal("cannot measure firing rate on zero frames")
    fired = sum(1 for row in array if fire_fn(row))
    return fired / array.shape[0]


def at_chance(
    firing_rate_on_noise: float, base_rate: float, *, tolerance: float = DEFAULT_NOISY_TV_TOLERANCE
) -> bool:

    for name, value in (("firing_rate_on_noise", firing_rate_on_noise), ("base_rate", base_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ControlRefusal(f"{name} must be a rate in [0, 1]")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0.0:
        raise ControlRefusal("tolerance must be a nonnegative number")
    return float(firing_rate_on_noise) <= float(base_rate) + float(tolerance)


@dataclass(frozen=True, slots=True)
class NoisyTvResult:

    firing_rate_on_noise: float
    base_rate: float
    tolerance: float
    n_noise_frames: int
    at_chance: bool
    schema: str = CONTROLS_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "firing_rate_on_noise": round(float(self.firing_rate_on_noise), 12),
            "base_rate": round(float(self.base_rate), 12),
            "tolerance": round(float(self.tolerance), 12),
            "n_noise_frames": self.n_noise_frames,
            "at_chance": self.at_chance,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def noisy_tv_probe(
    fire_fn: Callable[[np.ndarray], bool],
    base_rate: float,
    *,
    seed: int,
    n_noise_frames: int = 2000,
    tolerance: float = DEFAULT_NOISY_TV_TOLERANCE,
    d_feat: int = D_FEAT,
) -> NoisyTvResult:

    if isinstance(n_noise_frames, bool) or not isinstance(n_noise_frames, int) or n_noise_frames <= 0:
        raise ControlRefusal("n_noise_frames must be a positive integer")
    noise = pure_aleatoric_channel(seed, n_noise_frames, d_feat=d_feat)
    rate = firing_rate_on_frames(fire_fn, noise)
    verdict = at_chance(rate, base_rate, tolerance=tolerance)
    return NoisyTvResult(
        firing_rate_on_noise=rate,
        base_rate=float(base_rate),
        tolerance=float(tolerance),
        n_noise_frames=n_noise_frames,
        at_chance=verdict,
    )
