"""Component 4: the three honest controls.

Every gate ships with three separate, honest controls, each isolating a different failure mode. They
are implemented as plain firing-set producers so the harness can score them through the same sealed
referee as the candidate.

(a) rate-matched-random. Fires the SAME count K as a given candidate firing set, with the positions
    permuted uniformly at random and matched per clip and per seed, so the compute histogram is
    byte-identical. Any accuracy gap is then WHERE compute is spent, not how much. This is the SkipNet
    and BlockDrop RandomK control.

(b) best-single and always-on. always-on fires on every frame (K = F, the max-compute reference).
    best-single is a non-learned, val-tuned total-flux threshold: it reads a single scalar (the pooled
    spectral flux per frame) and fires above a threshold chosen on the val rooms. It is deterministic.
    Because onset and nuisance grains carry matched total flux in disjoint bands, a bare flux threshold
    trips on the nuisance and cannot reach the ceiling, which is the point of the reference.

(c) noisy-TV. Injects a pure-aleatoric channel (RND-style: a fixed, deterministic, randomly
    initialized target f*(o) of the observation, with no reducible structure) and measures the gate's
    firing rate on that noise. A gate rewarded by raw novelty chases the noisy TV and fires
    preferentially on it; the required check is that firing on pure noise stays at chance. Preferential
    firing on the injected noise FAILS.

This module trains nothing. It reuses the sealed referee for the one place a control needs a score (the
val tuning of best-single). House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.science.budget import ARM_ALWAYS_ON, ARM_BEST_SINGLE, ARM_RATE_MATCHED_RANDOM
from mop.substrate.events import canonical_sha256

from .featurizer import D_FEAT
from .referee import score_arm
from .schema import COLLAR_FRAMES

CONTROLS_SCHEMA = "mop-starss23-escs-controls/v1"

_RMR_NAMESPACE = "mop.beds.starss23.controls.rate_matched_random"
_NOISY_TV_NAMESPACE = "mop.beds.starss23.controls.noisy_tv"
_RND_TARGET_NAMESPACE = "mop.beds.starss23.controls.rnd_target"


def _child_seed(seed: int, key: str) -> int:
    """Return a uint32 child seed domain-separated by ``(seed, key)``.

    ``mop.seeding.derive_seed32`` deliberately passes an in-range seed through unchanged and ignores
    its namespace, so it cannot separate two clips that share a small integer seed. The controls need
    a stream that differs per clip and per namespace even at seed 0, so the full key is hashed here,
    mirroring derive_seed32's own recipe for oversized seeds.
    """

    payload = json.dumps(
        {"seed": int(seed), "key": str(key)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(b"mop-starss23-controls-v1\0" + payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)

# The default at-chance tolerance band for the noisy-TV firing-rate check. A gate may fire on pure noise
# no more often than its base rate plus this slack; firing preferentially above that band is a failure.
DEFAULT_NOISY_TV_TOLERANCE = 0.05


class ControlRefusal(ValueError):
    """Raised when a control input is malformed or a control invariant would be violated."""


def _unique_sorted_frames(frames: Sequence[int], n_frames: int, label: str) -> list[int]:
    seen: set[int] = set()
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise ControlRefusal(f"{label} must contain frames in [0, {n_frames})")
        seen.add(frame)
    return sorted(seen)


# ---------------------------------------------------------------------------
# (a) rate-matched-random: same firing COUNT, positions permuted, matched per clip and per seed.
# ---------------------------------------------------------------------------


def rate_matched_random_fires(
    candidate_fires: Sequence[int], n_frames: int, *, seed: int, clip_id: str
) -> list[int]:
    """Return K firings permuted uniformly at random, where K is the candidate's firing count.

    Matched per clip and per seed: the random positions are drawn from a child seed derived from both
    ``seed`` and ``clip_id``, so the control is reproducible and the firing COUNT is byte-equal to the
    candidate's on that clip and seed. The positions are otherwise uniform over the clip's frames, so
    the only difference from the candidate is WHERE the firings land.
    """

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ControlRefusal("clip_id must be a non-empty string")
    fires = _unique_sorted_frames(candidate_fires, n_frames, "candidate_fires")
    k = len(fires)
    if k >= n_frames:
        return list(range(n_frames))
    rng = np.random.default_rng(_child_seed(seed, f"{_RMR_NAMESPACE}:{clip_id}"))
    positions = rng.choice(n_frames, size=k, replace=False)
    return sorted(int(position) for position in positions)


@dataclass(frozen=True, slots=True)
class RateMatchedRandomControl:
    """A per-seed rate-matched-random control: same count as the candidate, positions permuted."""

    seed: int
    arm_kind: str = ARM_RATE_MATCHED_RANDOM

    def fires_for_clip(self, candidate_fires: Sequence[int], n_frames: int, clip_id: str) -> list[int]:
        return rate_matched_random_fires(
            candidate_fires, n_frames, seed=self.seed, clip_id=clip_id
        )


# ---------------------------------------------------------------------------
# (b) always-on and best-single (a non-learned, val-tuned total-flux threshold).
# ---------------------------------------------------------------------------


def always_on_fires(n_frames: int) -> list[int]:
    """Always-on reference: fire on every frame. K = F, the maximum-compute arm. Deterministic."""

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return list(range(n_frames))


def never_update_reestimates(n_frames: int) -> list[int]:
    """The shared zero-compute floor: never re-estimate and coast from the initial value."""

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    return []


def frame_flux(features: np.ndarray) -> np.ndarray:
    """The single scalar the best-single reference reads: pooled half-wave-rectified flux per frame."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != D_FEAT:
        raise ControlRefusal(f"features must be shape (n_frames, {D_FEAT})")
    return array.sum(axis=1)


def best_single_fires(flux: np.ndarray, threshold: float) -> list[int]:
    """Fire on every frame whose pooled flux is at least the threshold. Deterministic."""

    array = np.asarray(flux, dtype=np.float64)
    if array.ndim != 1:
        raise ControlRefusal("flux must be a 1-D per-frame array")
    return [int(index) for index in np.nonzero(array >= float(threshold))[0]]


def _candidate_thresholds(flux_values: np.ndarray) -> list[float]:
    """Deterministic candidate thresholds: the midpoints between sorted unique flux values, plus a top."""

    unique = np.unique(flux_values)
    if unique.size == 0:
        return [0.0]
    midpoints = ((unique[:-1] + unique[1:]) / 2.0).tolist() if unique.size > 1 else []
    # A threshold above the maximum fires nothing; one at or below the minimum fires everything.
    return [float(unique[0] - 1.0), *[float(value) for value in midpoints], float(unique[-1] + 1.0)]


def tune_best_single_threshold(
    val_clips: Sequence[tuple[np.ndarray, Sequence[int]]], *, collar: int = COLLAR_FRAMES
) -> float:
    """Pick the total-flux threshold that maximizes pooled val onset F1. Non-learned and deterministic.

    ``val_clips`` is a sequence of ``(features, gt_frames)`` over the val rooms. The threshold sweep is
    the set of midpoints between sorted unique flux values pooled across the val clips. Ties on F1 are
    broken toward the HIGHER threshold, which fires less and keeps the reference conservative.
    """

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
    """A non-learned best-single reference: a val-tuned total-flux threshold, applied deterministically."""

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
    """The always-on reference. Fires on every frame; the maximum-compute arm."""

    arm_kind: str = ARM_ALWAYS_ON

    def fires_for_clip(self, n_frames: int) -> list[int]:
        return always_on_fires(n_frames)


# ---------------------------------------------------------------------------
# (c) noisy-TV: an injected pure-aleatoric channel with a fixed RND target and an at-chance check.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RndTarget:
    """A fixed, deterministic, randomly initialized target f*(o) in the style of Random Network Distillation.

    The map is seeded once and never trained, so it is byte-reproducible. ``novelty`` returns the norm
    of f*(o) per frame: a gate rewarded by raw prediction error against an untrained predictor (the zero
    map) chases exactly this quantity, and pure aleatoric input drives it high, which is the noisy-TV
    trap. Nothing here is trained; the target is a fixed random projection.
    """

    seed: int
    d_in: int = D_FEAT
    d_out: int = 32

    def _matrix(self) -> np.ndarray:
        rng = np.random.default_rng(_child_seed(self.seed, _RND_TARGET_NAMESPACE))
        return rng.standard_normal((self.d_out, self.d_in)) / np.sqrt(self.d_in)

    def novelty(self, features: np.ndarray) -> np.ndarray:
        """Raw novelty ||f*(o)|| per frame. Deterministic in the seed."""

        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.d_in:
            raise ControlRefusal(f"features must be shape (n_frames, {self.d_in})")
        projected = array @ self._matrix().T
        return np.sqrt((projected * projected).sum(axis=1))


def pure_aleatoric_channel(seed: int, n_frames: int, *, d_feat: int = D_FEAT) -> np.ndarray:
    """A block of pure-aleatoric feature rows with no reducible structure. Deterministic in the seed."""

    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise ControlRefusal("n_frames must be a positive integer")
    rng = np.random.default_rng(_child_seed(seed, _NOISY_TV_NAMESPACE))
    return rng.standard_normal((n_frames, d_feat))


def noise_chasing_fire_fn(
    normal_features: np.ndarray, *, base_rate: float, seed: int
) -> Callable[[np.ndarray], bool]:
    """Build a gate that fires on raw RND novelty at a threshold calibrated to base_rate on normal content.

    On normal content this fires at approximately ``base_rate``; on pure aleatoric input, whose raw
    novelty is larger, it fires preferentially above base_rate. It is the honest illustration of the
    noisy-TV failure a real gate must NOT exhibit.
    """

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
    """The fraction of frames a gate fires on. Used to measure firing on the injected noise channel."""

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
    """Return whether firing on the injected noise stayed at chance (no preferential firing).

    Passing means the noise firing rate does not exceed the base rate by more than the tolerance. Firing
    LESS on noise than on normal content is honest and passes; firing preferentially MORE fails.
    """

    for name, value in (("firing_rate_on_noise", firing_rate_on_noise), ("base_rate", base_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ControlRefusal(f"{name} must be a rate in [0, 1]")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0.0:
        raise ControlRefusal("tolerance must be a nonnegative number")
    return float(firing_rate_on_noise) <= float(base_rate) + float(tolerance)


@dataclass(frozen=True, slots=True)
class NoisyTvResult:
    """The outcome of the noisy-TV control: the firing rate on injected noise and the at-chance verdict."""

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
    """Inject a pure-aleatoric channel, measure the gate's firing rate on it, and check for at-chance.

    ``fire_fn`` maps one feature row to a fire decision. ``base_rate`` is the gate's firing rate on
    normal content. The probe generates ``n_noise_frames`` pure-noise frames, measures how often the
    gate fires on them, and requires that rate to stay at chance. Preferential firing on the noise
    fails, which is the required noisy-TV guard.
    """

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
