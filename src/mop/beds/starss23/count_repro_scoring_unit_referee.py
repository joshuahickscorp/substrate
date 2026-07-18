"""Scoring-unit adversarial reproduction, component A: the clip-macro count referee.

This is a net-new, additive module for the STARSS23 concurrent-source-counting bed. It re-scores the
exact same coasted count estimates as the sealed pooled referee, but with the CLIP as the experimental
unit instead of the pooled frame micro-average. It kills the spurious win that would arise from
pseudoreplication inflation: in the pooled micro-average, frames within a clip are correlated and long
clips dominate the pool, so a win concentrated in one long clip can masquerade as a corpus-wide effect.
The sealed prereg itself names the clip as the experimental unit and refuses a clip bootstrap; this
reproduction honours that by re-scoring with the clip as the unit.

Nothing sealed is edited. Coasting is reused unchanged BY IMPORT from the sealed ``count_referee``
(``coast_emitted`` and ``mae_clip``), so the per-clip absolute error and the per-clip frame count are
byte-identical to the pooled path. Only the AGGREGATION differs:

    MAE_clip(c) = abs_error(c) / n_frames(c)                 (per-clip mean absolute error)
    macro_MAE   = (1 / n_clips) * sum_c MAE_clip(c)          (every clip weighted equally)

replacing the pooled frame micro-average ``sum_c abs_error(c) / sum_c n_frames(c)``. This macro-MAE is a
nonnegative minimized float, so it feeds straight into the reused ``CountArm``/``CountArmSeedResult`` and
``run_matched_budget`` machinery, which is metric-agnostic between the micro and the macro pooling.

The module also provides the honest "clip is the experimental unit" corroborating test the sealed prereg
deferred: an EXACT sign-flip permutation OVER CLIPS. With about twenty test clips a full 2^n_clips
enumeration is done by meet in the middle so it stays exact and fast, and the reused five-seed sign-flip
enumeration (n = 5, 2^5 = 32) is left untouched for the primary statistic.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

from .count_referee import COLD_START, mae_clip  # reused unchanged from the sealed pooled referee

COUNT_REPRO_SCORING_UNIT_REFEREE_SCHEMA = "mop-starss23-count-repro-scoring-unit-referee/v1"
SCORING_UNIT = "clip-macro"
MACRO_METRIC_RULE = (
    "coasted-count-MAE re-scored with the clip as the experimental unit: MAE_clip = abs_error / n_frames, "
    "arm score = clip-macro mean (each clip weighted equally), replacing the pooled frame micro-average"
)

# The tie tolerance for counting a sign assignment as reaching the observed statistic. It matches the
# sealed stats.exact_sign_flip convention so the two enumerations agree on genuine ties.
_TIE_EPS = 1e-9


class CountReproScoringUnitRefereeRefusal(ValueError):
    """Raised when a clip triple or a clip-cluster permutation input violates the macro-referee contract."""


@dataclass(frozen=True, slots=True)
class MacroClipScore:
    """One clip's coasted-count score under the clip-macro rule: absolute error, frame count, per-clip MAE."""

    clip_id: str
    abs_error_sum: int
    n_frames: int
    mae: float

    def payload(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "abs_error_sum": self.abs_error_sum,
            "n_frames": self.n_frames,
            "mae": round(float(self.mae), 12),
        }


@dataclass(frozen=True, slots=True)
class MacroCountScore:
    """A clip-macro arm score: the per-clip MAEs and the equal-weight clip-macro mean over them."""

    n_clips: int
    macro_mae: float
    per_clip: tuple[MacroClipScore, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": COUNT_REPRO_SCORING_UNIT_REFEREE_SCHEMA,
            "scoring_unit": SCORING_UNIT,
            "metric_rule": MACRO_METRIC_RULE,
            "n_clips": self.n_clips,
            "macro_mae": round(float(self.macro_mae), 12),
            "per_clip": {score.clip_id: score.payload() for score in self.per_clip},
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def clip_mae(self) -> dict[str, float]:
        return {score.clip_id: float(score.mae) for score in self.per_clip}


def macro_score_arm(
    clips: Iterable[tuple[str, Sequence[int], Sequence[int], Sequence[int]]],
    cold_start: int = COLD_START,
) -> MacroCountScore:
    """Score one arm with the clip as the experimental unit.

    ``clips`` is an iterable of ``(clip_id, gt_track, estimator_track, reestimate_frames)``. Per clip the
    reused sealed ``mae_clip`` re-coasts the estimator at the arm's re-estimation frames and returns the
    byte-identical ``(abs_error_sum, n_frames)``; the per-clip MAE is their ratio and the arm score is the
    equal-weight mean of the per-clip MAEs. A clip must carry at least one frame.
    """

    per_clip: list[MacroClipScore] = []
    seen: set[str] = set()
    for clip_id, gt_track, estimator_track, reestimate_frames in clips:
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise CountReproScoringUnitRefereeRefusal("clip_id must be a nonempty string")
        if clip_id in seen:
            raise CountReproScoringUnitRefereeRefusal(f"clip {clip_id!r} appears twice in one arm")
        seen.add(clip_id)
        abs_error_sum, n_frames = mae_clip(gt_track, estimator_track, reestimate_frames, cold_start)
        if n_frames <= 0:
            raise CountReproScoringUnitRefereeRefusal(f"clip {clip_id!r} has no frames to score")
        per_clip.append(
            MacroClipScore(
                clip_id=clip_id,
                abs_error_sum=abs_error_sum,
                n_frames=n_frames,
                mae=abs_error_sum / n_frames,
            )
        )
    if not per_clip:
        raise CountReproScoringUnitRefereeRefusal("a clip-macro arm needs at least one clip")
    macro_mae = sum(score.mae for score in per_clip) / len(per_clip)
    return MacroCountScore(n_clips=len(per_clip), macro_mae=macro_mae, per_clip=tuple(per_clip))


# ---------------------------------------------------------------------------
# Exact sign-flip permutation over the clips (the clip-clustered corroborating readout).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipClusterPermutation:
    """The exact clip-clustered sign-flip: one sign per clip over all 2^n_clips assignments."""

    n_clips: int
    permutations: int
    t_observed: float
    mean_delta: float
    n_clips_candidate_lower: int
    fraction_candidate_lower: float
    one_sided_p: float
    min_one_sided_p: float
    direction_agrees: bool

    def payload(self) -> dict[str, Any]:
        return {
            "n_clips": self.n_clips,
            "permutations": self.permutations,
            "t_observed": round(float(self.t_observed), 12),
            "mean_delta": round(float(self.mean_delta), 12),
            "n_clips_candidate_lower": self.n_clips_candidate_lower,
            "fraction_candidate_lower": round(float(self.fraction_candidate_lower), 12),
            "one_sided_p": round(float(self.one_sided_p), 12),
            "min_one_sided_p": round(float(self.min_one_sided_p), 12),
            "direction_agrees": self.direction_agrees,
            "unit": "clip",
            "statistic": "sum of per-clip paired deltas (rate_matched_random minus candidate)",
        }


def _exact_sign_flip_one_sided_meet_in_middle(deltas: Sequence[float]) -> tuple[float, float, int]:
    """Return ``(t_observed, one_sided_p, permutations)`` for the exact sign-flip over ``deltas``.

    The statistic is ``T(s) = sum_i s_i * delta_i`` with ``s_i`` in ``{+1, -1}``; the observed assignment
    is all plus, ``T_obs = sum_i delta_i``. The one-sided upper-tail p is the fraction of the ``2^n`` sign
    assignments with ``T(s) >= T_obs``. This is a subset-sum threshold count, so it is done exactly by meet
    in the middle: enumerate one half's partial sums, sort them, and binary-search the complementary
    threshold for each partial sum of the other half. Exact and deterministic for any ``n``.
    """

    values = [float(v) for v in deltas]
    n = len(values)
    if n == 0:
        raise CountReproScoringUnitRefereeRefusal("the clip sign-flip needs at least one paired delta")
    t_observed = sum(values)
    half = n // 2
    left = values[:half]
    right = values[half:]

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
        # Count right-half assignments with left_sum + right_sum >= T_obs - eps.
        cutoff = threshold_base - left_sum
        count += len(right_sums) - bisect_left(right_sums, cutoff)
    permutations = 2**n
    return t_observed, count / permutations, permutations


def exact_sign_flip_over_clips(deltas: Sequence[float]) -> ClipClusterPermutation:
    """Run the exact clip-clustered sign-flip over per-clip paired deltas (rate_matched_random - candidate).

    A positive per-clip delta means the candidate reached a strictly lower per-clip MAE on that clip. The
    direction agrees with the pooled and macro claim only when the mean per-clip delta is strictly positive
    (the candidate is lower on the clip average). Reports the fraction of clips on which the candidate is
    strictly lower and the exact one-sided permutation p over signs assigned per clip.
    """

    values = [float(v) for v in deltas]
    n = len(values)
    if n == 0:
        raise CountReproScoringUnitRefereeRefusal("the clip sign-flip needs at least one paired delta")
    t_observed, one_sided_p, permutations = _exact_sign_flip_one_sided_meet_in_middle(values)
    n_lower = sum(1 for value in values if value > 0.0)
    mean_delta = t_observed / n
    return ClipClusterPermutation(
        n_clips=n,
        permutations=permutations,
        t_observed=t_observed,
        mean_delta=mean_delta,
        n_clips_candidate_lower=n_lower,
        fraction_candidate_lower=n_lower / n,
        one_sided_p=one_sided_p,
        min_one_sided_p=1.0 / permutations,
        direction_agrees=mean_delta > 0.0,
    )
