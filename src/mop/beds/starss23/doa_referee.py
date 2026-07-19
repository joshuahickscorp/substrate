
from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mop.substrate.events import canonical_sha256

from .doa_labels import great_circle_degrees_batch

DOA_REFEREE_SCHEMA = "mop-starss23-doa-referee/v1"

DOA_COLD_START_AZIMUTH_DEG = 0.0
DOA_COLD_START_ELEVATION_DEG = 0.0
DOA_COLD_START = (DOA_COLD_START_AZIMUTH_DEG, DOA_COLD_START_ELEVATION_DEG)

METRIC_RULE = (
    "coasted great-circle DoA error: emitted holds the last re-estimated direction, cold-start boresight "
    "(0, 0), clip-macro (equal-weight per-clip mean) is PRIMARY; pooled frame micro-average is secondary"
)

_MAX_CLIP_ENUMERATION = 40
_TIE_EPS = 1e-9


class DoaRefereeRefusal(ValueError):
    pass


def _require_directions(track: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(track, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise DoaRefereeRefusal(f"{label} must be shape (n_frames, 2)")
    return array


def _require_reestimate_frames(frames: Sequence[int], n_frames: int) -> tuple[int, ...]:
    prepared: list[int] = []
    previous = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise DoaRefereeRefusal(f"reestimate_frames must lie in [0, {n_frames})")
        if frame <= previous:
            raise DoaRefereeRefusal("reestimate_frames must be strictly sorted and unique")
        previous = frame
        prepared.append(frame)
    return tuple(prepared)




def coast_emitted_direction(
    estimator_track: np.ndarray,
    reestimate_frames: Sequence[int],
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> np.ndarray:

    estimator = _require_directions(estimator_track, "estimator_track")
    n_frames = estimator.shape[0]
    if n_frames == 0:
        raise DoaRefereeRefusal("estimator_track must be nonempty")
    reestimates = _require_reestimate_frames(reestimate_frames, n_frames)
    reestimate_set = set(reestimates)
    emitted = np.empty((n_frames, 2), dtype=np.float64)
    current = np.array(cold_start, dtype=np.float64)
    for t in range(n_frames):
        if t in reestimate_set:
            current = estimator[t]
        emitted[t] = current
    return emitted


def angular_error_track(
    gt_directions: np.ndarray, emitted_directions: np.ndarray, active_mask: np.ndarray
) -> np.ndarray:

    gt = _require_directions(gt_directions, "gt_directions")
    emitted = _require_directions(emitted_directions, "emitted_directions")
    mask = np.asarray(active_mask, dtype=bool)
    if gt.shape[0] != emitted.shape[0] or gt.shape[0] != mask.shape[0]:
        raise DoaRefereeRefusal("gt_directions, emitted_directions, and active_mask must share n_frames")
    errors = great_circle_degrees_batch(gt, emitted)
    return errors[mask]




def mae_deg_clip(
    gt_directions: np.ndarray,
    estimator_track: np.ndarray,
    reestimate_frames: Sequence[int],
    active_mask: np.ndarray,
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> tuple[float, int]:

    emitted = coast_emitted_direction(estimator_track, reestimate_frames, cold_start)
    errors = angular_error_track(gt_directions, emitted, active_mask)
    n_active = int(errors.shape[0])
    if n_active == 0:
        raise DoaRefereeRefusal("a clip with no active frame cannot be scored")
    return float(errors.mean()), n_active


@dataclass(frozen=True, slots=True)
class MacroClipDoaScore:

    clip_id: str
    mae_deg: float
    n_active_frames: int

    def payload(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "mae_deg": round(float(self.mae_deg), 12),
            "n_active_frames": self.n_active_frames,
        }


@dataclass(frozen=True, slots=True)
class MacroDoaScore:

    n_clips: int
    macro_mae_deg: float
    per_clip: tuple[MacroClipDoaScore, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": DOA_REFEREE_SCHEMA,
            "scoring_unit": "clip-macro",
            "metric_rule": METRIC_RULE,
            "n_clips": self.n_clips,
            "macro_mae_deg": round(float(self.macro_mae_deg), 12),
            "per_clip": {score.clip_id: score.payload() for score in self.per_clip},
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def clip_mae(self) -> dict[str, float]:
        return {score.clip_id: float(score.mae_deg) for score in self.per_clip}


def macro_score_arm(
    clips: Iterable[tuple[str, np.ndarray, np.ndarray, Sequence[int], np.ndarray]],
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> MacroDoaScore:

    per_clip: list[MacroClipDoaScore] = []
    seen: set[str] = set()
    for clip_id, gt, estimator, reestimate_frames, active_mask in clips:
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise DoaRefereeRefusal("clip_id must be a nonempty string")
        if clip_id in seen:
            raise DoaRefereeRefusal(f"clip {clip_id!r} appears twice in one arm")
        seen.add(clip_id)
        mae, n_active = mae_deg_clip(gt, estimator, reestimate_frames, active_mask, cold_start)
        per_clip.append(MacroClipDoaScore(clip_id=clip_id, mae_deg=mae, n_active_frames=n_active))
    if not per_clip:
        raise DoaRefereeRefusal("a clip-macro arm needs at least one clip")
    macro_mae = math.fsum(score.mae_deg for score in per_clip) / len(per_clip)
    return MacroDoaScore(n_clips=len(per_clip), macro_mae_deg=macro_mae, per_clip=tuple(per_clip))




@dataclass(frozen=True, slots=True)
class PooledDoaScore:

    n_frames: int
    pooled_mae_deg: float

    def payload(self) -> dict[str, Any]:
        return {
            "schema": DOA_REFEREE_SCHEMA,
            "scoring_unit": "pooled-frame (secondary, not the survive criterion)",
            "n_frames": self.n_frames,
            "pooled_mae_deg": round(float(self.pooled_mae_deg), 12),
        }


def pooled_score_arm(
    clips: Iterable[tuple[str, np.ndarray, np.ndarray, Sequence[int], np.ndarray]],
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> PooledDoaScore:

    total_error = 0.0
    total_frames = 0
    for _clip_id, gt, estimator, reestimate_frames, active_mask in clips:
        emitted = coast_emitted_direction(estimator, reestimate_frames, cold_start)
        errors = angular_error_track(gt, emitted, active_mask)
        total_error += float(errors.sum())
        total_frames += int(errors.shape[0])
    if total_frames == 0:
        raise DoaRefereeRefusal("pooled_score_arm needs at least one active frame across all clips")
    return PooledDoaScore(n_frames=total_frames, pooled_mae_deg=total_error / total_frames)




@dataclass(frozen=True, slots=True)
class ClipSignFlipResult:

    n_clips: int
    permutations: int
    t_observed: float
    mean_delta: float
    n_clips_favorable: int
    fraction_favorable: float
    one_sided_p: float
    min_one_sided_p: float
    alpha: float
    one_sided_significant: bool

    def payload(self) -> dict[str, Any]:
        return {
            "unit": "clip",
            "n_clips": self.n_clips,
            "permutations": self.permutations,
            "t_observed": round(float(self.t_observed), 12),
            "mean_delta": round(float(self.mean_delta), 12),
            "n_clips_favorable": self.n_clips_favorable,
            "fraction_favorable": round(float(self.fraction_favorable), 12),
            "one_sided_p": round(float(self.one_sided_p), 15),
            "min_one_sided_p": round(float(self.min_one_sided_p), 15),
            "alpha": self.alpha,
            "one_sided_significant": self.one_sided_significant,
            "statistic": "sum of per-clip paired deltas (rate_matched_random minus candidate, seed-averaged)",
        }


def _exact_sign_flip_one_sided_meet_in_middle(deltas: Sequence[float]) -> tuple[float, float, int]:

    values = [float(v) for v in deltas]
    n = len(values)
    if n == 0:
        raise DoaRefereeRefusal("the clip sign-flip needs at least one paired delta")
    if n > _MAX_CLIP_ENUMERATION:
        raise DoaRefereeRefusal(
            f"exact clip enumeration is capped at n={_MAX_CLIP_ENUMERATION}; got n={n}. A documented Monte "
            "Carlo permutation fallback with the Phipson-Smyth (b+1)/(m+1) correction would be required "
            "above this cap; it is not implemented because the real subset anchors at n_test_clips = 21"
        )
    t_observed = math.fsum(values)
    half = n // 2
    left, right = values[:half], values[half:]

    def partial_sums(part: list[float]) -> list[float]:
        sums = [0.0]
        for value in part:
            nxt: list[float] = []
            for acc in sums:
                nxt.append(acc + value)
                nxt.append(acc - value)
            sums = nxt
        return sums

    right_sums = sorted(partial_sums(right))
    threshold_base = t_observed - _TIE_EPS
    count = 0
    for left_sum in partial_sums(left):
        cutoff = threshold_base - left_sum
        count += len(right_sums) - bisect_left(right_sums, cutoff)
    permutations = 2**n
    return t_observed, count / permutations, permutations


def exact_sign_flip_over_clips(deltas: Sequence[float], alpha: float = 0.05) -> ClipSignFlipResult:

    values = [float(v) for v in deltas]
    n = len(values)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise DoaRefereeRefusal("alpha must be a probability strictly between 0 and 1")
    t_observed, one_sided_p, permutations = _exact_sign_flip_one_sided_meet_in_middle(values)
    n_favorable = sum(1 for value in values if value > 0.0)
    mean_delta = t_observed / n
    return ClipSignFlipResult(
        n_clips=n,
        permutations=permutations,
        t_observed=t_observed,
        mean_delta=mean_delta,
        n_clips_favorable=n_favorable,
        fraction_favorable=n_favorable / n,
        one_sided_p=one_sided_p,
        min_one_sided_p=1.0 / permutations,
        alpha=float(alpha),
        one_sided_significant=one_sided_p <= float(alpha),
    )




@dataclass(frozen=True, slots=True)
class RoomMajorityResult:

    n_rooms: int
    permutations: int
    n_rooms_favorable: int
    one_sided_p: float
    min_one_sided_p: float
    per_room: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "unit": "room (majority vote of within-room clips)",
            "n_rooms": self.n_rooms,
            "permutations": self.permutations,
            "n_rooms_favorable": self.n_rooms_favorable,
            "one_sided_p": round(float(self.one_sided_p), 15),
            "min_one_sided_p": round(float(self.min_one_sided_p), 15),
            "per_room": list(self.per_room),
            "statistic": "count of rooms whose within-room clips strictly-majority-favor the candidate",
        }


def room_majority_collapse(clip_deltas_by_room: Mapping[str, Sequence[float]]) -> RoomMajorityResult:

    if not clip_deltas_by_room:
        raise DoaRefereeRefusal("room_majority_collapse needs at least one room")
    n_favorable = 0
    per_room: list[dict[str, Any]] = []
    for room_id in sorted(clip_deltas_by_room):
        deltas = [float(v) for v in clip_deltas_by_room[room_id]]
        if not deltas:
            raise DoaRefereeRefusal(f"room {room_id!r} has no clip deltas")
        n_pos = sum(1 for value in deltas if value > 0.0)
        favorable = n_pos * 2 > len(deltas)
        if favorable:
            n_favorable += 1
        per_room.append(
            {
                "room_id": room_id,
                "n_clips": len(deltas),
                "n_favorable_clips": n_pos,
                "favorable": favorable,
            }
        )
    n_rooms = len(per_room)
    permutations = 2**n_rooms
    one_sided_p = sum(math.comb(n_rooms, k) for k in range(n_favorable, n_rooms + 1)) / permutations
    return RoomMajorityResult(
        n_rooms=n_rooms,
        permutations=permutations,
        n_rooms_favorable=n_favorable,
        one_sided_p=one_sided_p,
        min_one_sided_p=1.0 / permutations,
        per_room=tuple(per_room),
    )
