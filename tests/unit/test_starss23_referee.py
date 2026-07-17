"""Tests for component 5, the sealed onset-F1 referee.

These pin the exact hand-computable F1 on known events, prove the strict one-to-one matcher resists the
inflation that point-adjustment suffers, and lock the collar boundary, nearest-first tie-break, micro
pooling, validation, and seal determinism.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import random

import pytest

from mop.beds.starss23.referee import (
    COLLAR_FRAMES,
    COLLAR_MS,
    MATCH_RULE,
    OnsetScore,
    RefereeRefusal,
    greedy_match,
    score_arm,
    score_clip,
    sealed_arm_report,
)


def test_collar_is_two_hundred_ms_on_the_hundred_ms_grid() -> None:
    assert COLLAR_FRAMES == 2
    assert COLLAR_MS == 200


def test_oracle_detection_scores_f1_one() -> None:
    gt = [10, 20, 30, 40]
    score = score_arm([(gt, list(gt))])
    assert (score.tp, score.fp, score.fn) == (4, 0, 0)
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_one_miss_one_false_positive_is_exactly_three_quarters() -> None:
    gt = [10, 20, 30, 40]
    pred = [10, 20, 30, 55]  # 40 missed, 55 spurious
    tp, fp, fn = score_clip(gt, pred)
    assert (tp, fp, fn) == (3, 1, 1)
    score = OnsetScore.from_counts(tp, fp, fn)
    assert score.precision == 0.75
    assert score.recall == 0.75
    assert score.f1 == 0.75


def test_collar_boundary_includes_two_frames_excludes_three() -> None:
    assert score_clip([10], [12]) == (1, 0, 0)  # abs(12 - 10) == 2 hits
    assert score_clip([10], [13]) == (0, 1, 1)  # abs(13 - 10) == 3 misses


def test_one_to_one_denies_the_flood_that_point_adjustment_rewards() -> None:
    # A four-fire flood onto a single onset. Strict one-to-one binds one and charges three false
    # positives, so F1 is 0.4, not the 1.0 a point-adjusted score would hand out.
    gt = [10]
    pred = [9, 10, 11, 12]
    tp, fp, fn = score_clip(gt, pred)
    assert (tp, fp, fn) == (1, 3, 0)
    assert OnsetScore.from_counts(tp, fp, fn).f1 == pytest.approx(0.4)


def test_nearest_first_binds_the_closest_ground_truth() -> None:
    # A single fire at 11 is within the collar of both 10 (distance 1) and 13 (distance 2). Nearest
    # first must bind it to 10 and leave 13 as a false negative.
    result = greedy_match([10, 13], [11])
    assert result.matched == ((10, 11),)
    assert result.unmatched_gt == (13,)
    assert result.unmatched_pred == ()


def test_micro_pooling_sums_counts_before_the_ratio() -> None:
    # Clip one is perfect (1 tp). Clip two detects one of two onsets (1 tp, 1 fn). Pooled: tp 2, fn 1.
    clips = [([10], [10]), ([20, 30], [20])]
    score = score_arm(clips)
    assert (score.tp, score.fp, score.fn) == (2, 0, 1)
    assert score.precision == 1.0
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(0.8)


def test_empty_clip_collapses_to_zero_f1() -> None:
    score = score_arm([([], [])])
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    assert score.f1 == 0.0


def test_input_order_does_not_change_the_result() -> None:
    a = score_clip([10, 20, 30], [30, 10, 20])
    b = score_clip([30, 20, 10], [10, 30, 20])
    assert a == b == (3, 0, 0)


def test_referee_refuses_bad_frames_and_collars() -> None:
    with pytest.raises(RefereeRefusal):
        score_clip([-1], [0])
    with pytest.raises(RefereeRefusal):
        score_clip([True], [0])  # bool is not a valid frame
    with pytest.raises(RefereeRefusal):
        greedy_match([0], [0], collar_frames=-1)
    with pytest.raises(RefereeRefusal):
        OnsetScore.from_counts(-1, 0, 0)


def test_sealed_report_is_deterministic_and_records_the_rule() -> None:
    # 35 is five frames from 30, past the two-frame collar, so this arm is genuinely imperfect.
    clips = [([10, 20, 30], [10, 20, 35])]
    first = sealed_arm_report(clips)
    second = sealed_arm_report(clips)
    assert first == second
    assert first["match_rule"] == MATCH_RULE
    assert first["collar_frames"] == 2
    assert first["score"]["f1"] != 1.0
    # The stored digest covers the true body, so raising the sealed F1 no longer matches a re-hash.
    tampered = dict(first)
    tampered["score"] = dict(tampered["score"], f1=1.0)
    assert tampered["digest"] == first["digest"]  # stale digest was carried over unchanged
    from mop.substrate.events import canonical_sha256

    honest = {k: v for k, v in tampered.items() if k != "digest"}
    assert canonical_sha256(honest) != tampered["digest"]


# ---------------------------------------------------------------------------
# The point-adjustment demonstration: strict PR resists an inflation PA suffers.
# These per-frame helpers live only in the test. The referee itself never point-adjusts.
# ---------------------------------------------------------------------------


def _segments(starts: list[int], length: int) -> list[range]:
    return [range(start, start + length) for start in starts]


def _strict_pointwise_f1(positives: set[int], flags: set[int]) -> float:
    tp = len(positives & flags)
    fp = len(flags - positives)
    fn = len(positives - flags)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _point_adjusted_f1(segments: list[range], flags: set[int], total: int) -> float:
    # Point-adjustment: any single flag inside a segment relabels the whole segment as detected, and
    # the extra flags inside it are forgiven. This is the protocol the bed forbids.
    positives = {frame for segment in segments for frame in segment}
    detected = {frame for segment in segments if flags & set(segment) for frame in segment}
    tp = len(detected)
    fp = len(flags - positives)
    fn = len(positives - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def test_point_adjustment_inflates_a_random_detector_but_strict_pr_does_not() -> None:
    total = 2000
    segments = _segments([100, 500, 900, 1300, 1700], length=100)
    positives = {frame for segment in segments for frame in segment}
    rng = random.Random(1234)
    flags = {frame for frame in range(total) if rng.random() < 0.10}

    strict = _strict_pointwise_f1(positives, flags)
    adjusted = _point_adjusted_f1(segments, flags, total)

    # The same random detector: near worthless under strict PR, near perfect under point-adjustment.
    assert strict < 0.30
    assert adjusted > 0.70
    assert adjusted - strict > 0.40


def test_referee_onset_scoring_is_not_inflatable_by_a_dense_flood() -> None:
    # The referee's own strict onset scoring on sparse ground truth against a dense random fire stream
    # stays low, because every extra fire is a false positive under one-to-one matching.
    gt = [100, 500, 900, 1300, 1700]
    rng = random.Random(99)
    fires = [frame for frame in range(2000) if rng.random() < 0.10]
    score = score_arm([(gt, fires)])
    assert score.f1 < 0.15
