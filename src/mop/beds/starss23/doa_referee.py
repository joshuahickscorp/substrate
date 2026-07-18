"""Direction-of-arrival bed, component 6: coasting, angular error, and the clip-macro primary referee.

This is the producer-side scorer and the single source of truth for grading a DoA re-estimation arm.
Coasting holds a direction instead of an integer count, directly mirroring ``count_referee.coast_emitted``:
the emitted track holds the last re-estimated direction, defaulting to a fixed, physically meaningless
boresight cold start (the direction-of-arrival analogue of the counting bed's ``COLD_START = 0``) so the
never-update floor arm is well defined. Every re-estimation, even one at frame 0, is spent from the
budget; no arm gets a free frame-0 look.

The PRIMARY statistic is clip-macro (equal-weight mean of per-clip mean angular error), never pooled-frame:
this directly targets the failure mode that killed the counting bed's ``scoring_unit`` reproduction (doc
28), where pooled-frame pseudoreplication let a long clip dominate the corpus-wide number. Pooled-frame is
retained only as an explicitly labeled secondary corroborating readout, never the survive criterion.

The PRIMARY significance test is an exact one-sided upper-tail sign-flip permutation over CLIP-LEVEL paired
deltas (meet in the middle, tractable well past the real ``n_test_clips``), re-implemented here rather than
imported from ``count_repro_scoring_unit_referee.py`` so this bed owns its own primary statistic instead of
borrowing a cross-bed private helper. The 5-seed sign-flip (``stats.exact_sign_flip``) is reused unchanged
as a REQUIRED-TO-AGREE-IN-DIRECTION secondary discipline.

An additional readout, ``room_majority_collapse``, is a documented departure from the literal design spec,
added directly from the calibration note's empirical finding: the real corpus's test clips are exactly
7 rooms times 3 clips each, so same-room clips share acoustics and are not fully exchangeable units. This
collapses each room's within-room clip deltas to a majority sign and runs an exact binomial-style sign-flip
over rooms (floor 1/2**n_rooms), and any clip-level result should be cross-checked against it before being
written into a doc, per the calibration note's stated bar.

House style: no em dashes and no en dashes.
"""

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

# An arbitrary but declared fixed boresight direction: the direction-of-arrival analogue of the counting
# bed's COLD_START = 0. It carries no physical meaning the way "zero sources" does for counting; it exists
# only so the never-update floor arm is well defined (a constant boresight direction forever, expected to
# be a genuinely bad floor). Also reused by doa_estimator.py as the silence value of the frozen estimator.
DOA_COLD_START_AZIMUTH_DEG = 0.0
DOA_COLD_START_ELEVATION_DEG = 0.0
DOA_COLD_START = (DOA_COLD_START_AZIMUTH_DEG, DOA_COLD_START_ELEVATION_DEG)

METRIC_RULE = (
    "coasted great-circle DoA error: emitted holds the last re-estimated direction, cold-start boresight "
    "(0, 0), clip-macro (equal-weight per-clip mean) is PRIMARY; pooled frame micro-average is secondary"
)

# The exact clip-level meet-in-the-middle enumeration is tractable to at least the low tens of clips
# (2**40 partial sums per half is still fast); above that it refuses rather than silently degrading to an
# approximation. This does not bind for the real run: the real subset anchors at n_test_clips = 21.
_MAX_CLIP_ENUMERATION = 40
_TIE_EPS = 1e-9


class DoaRefereeRefusal(ValueError):
    """Raised when a track, a re-estimation set, or a clip pairing violates the DoA referee contract."""


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


# ---------------------------------------------------------------------------
# Coasting and angular error.
# ---------------------------------------------------------------------------


def coast_emitted_direction(
    estimator_track: np.ndarray,
    reestimate_frames: Sequence[int],
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> np.ndarray:
    """Return the deterministic coasted emitted direction track: hold the last re-estimate, else cold_start.

    ``estimator_track`` is the frozen estimator output E at every frame, shape (n_frames, 2) degrees.
    ``reestimate_frames`` is the sorted unique subset R of frames the arm actually re-estimated. Strictly
    causal: emitted(t) depends only on E and R at frames <= t.
    """

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
    """Great-circle error per active frame only. Inactive frames are excluded: there is no ground truth."""

    gt = _require_directions(gt_directions, "gt_directions")
    emitted = _require_directions(emitted_directions, "emitted_directions")
    mask = np.asarray(active_mask, dtype=bool)
    if gt.shape[0] != emitted.shape[0] or gt.shape[0] != mask.shape[0]:
        raise DoaRefereeRefusal("gt_directions, emitted_directions, and active_mask must share n_frames")
    errors = great_circle_degrees_batch(gt, emitted)
    return errors[mask]


# ---------------------------------------------------------------------------
# PRIMARY: clip-macro (equal-weight per-clip mean).
# ---------------------------------------------------------------------------


def mae_deg_clip(
    gt_directions: np.ndarray,
    estimator_track: np.ndarray,
    reestimate_frames: Sequence[int],
    active_mask: np.ndarray,
    cold_start: tuple[float, float] = DOA_COLD_START,
) -> tuple[float, int]:
    """Return ``(mean_angular_error_deg, n_active_frames)`` for one clip. Refuses if no frame is active."""

    emitted = coast_emitted_direction(estimator_track, reestimate_frames, cold_start)
    errors = angular_error_track(gt_directions, emitted, active_mask)
    n_active = int(errors.shape[0])
    if n_active == 0:
        raise DoaRefereeRefusal("a clip with no active frame cannot be scored")
    return float(errors.mean()), n_active


@dataclass(frozen=True, slots=True)
class MacroClipDoaScore:
    """One clip's coasted DoA score under the clip-macro rule: mean angular error and active-frame count."""

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
    """A clip-macro arm score: PRIMARY. Equal-weight mean of per-clip mean angular errors."""

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
    """Score one arm with the clip as the experimental unit (PRIMARY).

    ``clips`` is an iterable of ``(clip_id, gt_directions, estimator_track, reestimate_frames,
    active_mask)``. The arm score is the equal-weight mean of the per-clip mean angular errors.
    """

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


# ---------------------------------------------------------------------------
# SECONDARY: pooled frame micro-average. Explicitly non-primary; never the survive criterion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PooledDoaScore:
    """A pooled frame micro-average arm score. SECONDARY, corroboration only, never the survive criterion."""

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
    """Micro-average an arm across clips: sum of errors over sum of active frames. SECONDARY readout only."""

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


# ---------------------------------------------------------------------------
# PRIMARY significance test: exact clip-level sign-flip, meet in the middle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipSignFlipResult:
    """The exact clip-level sign-flip permutation: one sign per test clip, PRIMARY significance test."""

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
    """Exact one-sided upper-tail sign-flip p over ``deltas`` by meet in the middle.

    The statistic is ``T(s) = sum_i s_i * delta_i`` with ``s_i`` in ``{+1, -1}``; the observed assignment
    is all-plus, ``T_obs = sum_i delta_i``. The one-sided p is the fraction of the ``2**n`` sign
    assignments with ``T(s) >= T_obs``. Enumerate one half's partial sums, sort them, and binary-search the
    complementary threshold for each partial sum of the other half: exact and tractable well past n=21.

    Re-implemented here rather than imported from ``count_repro_scoring_unit_referee.py`` so this bed owns
    its own primary statistic rather than borrowing a cross-bed private helper.
    """

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
    """Run the PRIMARY exact clip-level sign-flip over per-clip paired deltas.

    ``deltas[i]`` is the mean over the paired seeds of ``(MAE_clip_i(rate_matched_random) -
    MAE_clip_i(candidate))``, one entry per test clip; positive means the candidate is better on that
    clip's seed-averaged score.
    """

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


# ---------------------------------------------------------------------------
# Room-majority collapse: additive discipline from the calibration note's room-clustering finding.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoomMajorityResult:
    """Exact room-level sign-flip over the majority-vote collapse of each room's within-room clip deltas.

    Calibration-note discipline: the real test split is exactly 7 rooms times 3 clips, and same-room clips
    share acoustics, so the true number of independent exchangeable units may be closer to n_rooms than
    n_clips. Each room collapses to a binary "favorable" vote (a strict majority of its clips favor the
    candidate); the one-sided p is the exact binomial upper tail P(X >= n_favorable | Binomial(n_rooms, 1/2)).
    """

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
    """Collapse per-clip deltas to a per-room majority vote and run the exact room-level sign-flip.

    ``clip_deltas_by_room`` maps room id to the (seed-averaged) per-clip deltas of the clips in that room,
    using the same delta convention as ``exact_sign_flip_over_clips`` (positive favors the candidate).
    """

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
