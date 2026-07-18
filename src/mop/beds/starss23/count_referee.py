"""Concurrent-source-counting bed, component 5: the sealed coasted-count-MAE referee.

This is the producer-side scorer and the single source of truth for grading a count arm. It is the
analogue of the onset-F1 referee, but for a pointwise level metric it needs NO matching: there is nothing
to align, so the greedy one-to-one machinery is deliberately absent and the metric is deterministic by
construction. Lower MAE is better; a tie (equal MAE) is a null at the harness level.

Coasting
--------
A count estimate goes stale between re-estimations. The emitted track holds the last re-estimated value:

    emitted(t) = E[ max{ r in R : r <= t } ]      if any r in R with r <= t
    emitted(t) = c0                                otherwise

with the preregistered cold start ``c0 = COLD_START = 0`` (a silence prior). Every re-estimation, even one
at frame 0, is spent from the budget; no arm gets a free frame-0 look. ``emitted(t)`` depends only on E at
frames <= t and on R at frames <= t, so it is strictly causal with no leakage.

Metric
------
    MAE_clip = (1/T) * sum_t |emitted(t) - C_gt(t)|
    MAE_arm  = ( sum_clips sum_t |emitted(t) - C_gt(t)| ) / ( sum_clips T )

The pool is a frame micro-average across test clips (numerator and denominator summed over clips first),
exactly mirroring the onset referee's pooled counts. The estimator track E is a property of the clip and
is shared by every arm; only R differs per arm. Sealing one E per clip makes always-on's MAE equal
``mean(|E - C_gt|)`` and never-update's MAE equal ``mean(|C_gt|)``, and lets the independent verifier
cross-check every arm against the same E. This module performs evaluator-side arithmetic only, charges no
arm compute, and establishes no scientific claim.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

COUNT_REFEREE_SCHEMA = "mop-starss23-count-referee/v1"
COLD_START = 0
METRIC_RULE = (
    "coasted-count-MAE: emitted holds the last re-estimate, cold-start 0, pooled frame micro-average"
)


class CountRefereeRefusal(ValueError):
    """Raised when a track, a re-estimation set, or a clip pairing violates the sealed referee contract."""


def _require_int_track(track: Sequence[int], label: str) -> tuple[int, ...]:
    if not isinstance(track, (list, tuple)):
        raise CountRefereeRefusal(f"{label} must be a list or tuple of integers")
    out: list[int] = []
    for value in track:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CountRefereeRefusal(f"{label} must contain only nonnegative integers")
        out.append(value)
    return tuple(out)


def _require_reestimate_frames(frames: Sequence[int], n_frames: int) -> tuple[int, ...]:
    """Validate a re-estimation frame set: sorted, unique, and every frame in [0, n_frames)."""

    prepared: list[int] = []
    previous = -1
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= n_frames:
            raise CountRefereeRefusal(f"reestimate_frames must lie in [0, {n_frames})")
        if frame <= previous:
            raise CountRefereeRefusal("reestimate_frames must be strictly sorted and unique")
        previous = frame
        prepared.append(frame)
    return tuple(prepared)


def coast_emitted(
    estimator_track: Sequence[int],
    reestimate_frames: Sequence[int],
    cold_start: int = COLD_START,
) -> tuple[int, ...]:
    """Return the deterministic coasted emitted track: hold the last re-estimated value, else cold_start.

    ``estimator_track`` is the frozen estimator output E at every frame; ``reestimate_frames`` is the sorted
    unique subset R of frames the arm actually re-estimated. Strictly causal.
    """

    if isinstance(cold_start, bool) or not isinstance(cold_start, int) or cold_start < 0:
        raise CountRefereeRefusal("cold_start must be a nonnegative integer")
    estimator = _require_int_track(estimator_track, "estimator_track")
    n_frames = len(estimator)
    if n_frames == 0:
        raise CountRefereeRefusal("estimator_track must be nonempty")
    reestimates = _require_reestimate_frames(reestimate_frames, n_frames)
    reestimate_set = set(reestimates)
    emitted: list[int] = []
    current = cold_start
    for t in range(n_frames):
        if t in reestimate_set:
            current = estimator[t]
        emitted.append(current)
    return tuple(emitted)


def mae_clip(
    gt_track: Sequence[int],
    estimator_track: Sequence[int],
    reestimate_frames: Sequence[int],
    cold_start: int = COLD_START,
) -> tuple[int, int]:
    """Return ``(abs_error_sum, n_frames)`` for one clip so arms pool exactly. len(gt) must equal len(E)."""

    gt = _require_int_track(gt_track, "gt_track")
    estimator = _require_int_track(estimator_track, "estimator_track")
    if len(gt) != len(estimator):
        raise CountRefereeRefusal("gt_track and estimator_track must be the same length")
    emitted = coast_emitted(estimator, reestimate_frames, cold_start)
    abs_error_sum = sum(abs(emitted[t] - gt[t]) for t in range(len(gt)))
    return abs_error_sum, len(gt)


@dataclass(frozen=True, slots=True)
class CountScore:
    """A sealed coasted-count-MAE score: pooled absolute-error sum, frame count, and the MAE."""

    abs_error_sum: int
    n_frames: int
    mae: float

    @classmethod
    def from_pool(cls, abs_error_sum: int, n_frames: int) -> "CountScore":
        for name, value in (("abs_error_sum", abs_error_sum), ("n_frames", n_frames)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CountRefereeRefusal(f"CountScore.{name} must be a nonnegative integer")
        if n_frames == 0:
            raise CountRefereeRefusal("CountScore needs at least one pooled frame")
        return cls(abs_error_sum=abs_error_sum, n_frames=n_frames, mae=abs_error_sum / n_frames)

    def payload(self) -> dict[str, Any]:
        return {
            "abs_error_sum": self.abs_error_sum,
            "n_frames": self.n_frames,
            "mae": round(float(self.mae), 12),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def score_arm(
    clips: Iterable[tuple[Sequence[int], Sequence[int], Sequence[int]]],
    cold_start: int = COLD_START,
) -> CountScore:
    """Micro-average an arm across clips.

    ``clips`` is an iterable of ``(gt_track, estimator_track, reestimate_frames)`` per clip. Absolute error
    and frame counts are summed across clips and a single pooled MAE is computed from the totals.
    """

    abs_error_sum = 0
    n_frames = 0
    for gt_track, estimator_track, reestimate_frames in clips:
        clip_abs, clip_frames = mae_clip(gt_track, estimator_track, reestimate_frames, cold_start)
        abs_error_sum += clip_abs
        n_frames += clip_frames
    return CountScore.from_pool(abs_error_sum, n_frames)


def sealed_arm_report(
    clips: Iterable[tuple[Sequence[int], Sequence[int], Sequence[int]]],
    cold_start: int = COLD_START,
) -> dict[str, Any]:
    """Score an arm and wrap it in a byte-sealed referee block recording the exact rule and a digest."""

    score = score_arm(clips, cold_start)
    body = {
        "schema": COUNT_REFEREE_SCHEMA,
        "metric_rule": METRIC_RULE,
        "cold_start": cold_start,
        "score": score.payload(),
    }
    body["digest"] = canonical_sha256(body)
    return body
