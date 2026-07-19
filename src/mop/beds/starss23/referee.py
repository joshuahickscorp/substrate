"""Component 5: the sealed onset-F1 referee.

This is the producer-side scorer. It is the single source of truth for how a candidate or control arm
is graded, and its output is byte-sealed so an independent verifier (component 8b) can re-derive every
number from specification without importing this module.

Metric and why it was chosen
-----------------------------
The referee reports onset F1 at the DCASE plus or minus 200 ms collar. At the frozen STARSS23 grid of
100 ms label frames that collar is exactly two frames on each side, so a predicted onset frame p matches
a ground-truth onset frame g if and only if ``abs(p - g) <= 2``. The DCASE fixed-collar rule is chosen
over the Kinetics-GEBD relative-distance rule (hit iff ``abs(t_pred - t_gt) <= threshold * L``) because
onset latency is a fixed physical quantity: two hundred milliseconds of tolerance is the same whether
the clip is ten seconds or ten minutes, whereas GEBD's tolerance grows with instance length and would
reward a detector for being vague on long clips.

Matching
--------
Greedy one-to-one nearest-first. Ground-truth and predicted frame lists are sorted ascending, then every
in-collar pair (g, p) is ordered by (distance, gt_frame, pred_frame, gt_index, pred_index) ascending and
assigned greedily, taking a pair only when neither its ground-truth frame nor its predicted frame is
already matched. This makes the match a total deterministic function of the two input multisets, so a
separately authored verifier that implements the same written rule reproduces it byte for byte. Each
ground-truth onset can absorb at most one prediction and each prediction can satisfy at most one
ground-truth onset.

Strict point-wise precision and recall, never point-adjusted
------------------------------------------------------------
Counts are strict and point-wise. Every predicted onset that is not matched is a false positive and
every ground-truth onset that is not matched is a false negative. Point-adjustment (relabel a whole
ground-truth segment as detected once any single point inside it is flagged, and forgive the flood of
other flags) is forbidden: under point-adjustment a uniform-random detector reaches F1 around 0.96 on
standard beds and becomes indistinguishable from a trained model. The one-to-one constraint here is
exactly what denies that inflation, because a dense detector spends its extra fires as false positives.

Pooling
-------
Counts are pooled across the test clips (micro-average): true positives, false positives, and false
negatives are summed over clips first, and a single precision, recall, and F1 are computed from the
pooled counts. Undefined ratios collapse to zero (sed_eval convention): precision is zero when there are
no predictions, recall is zero when there are no ground-truth onsets, and F1 is zero when precision plus
recall is zero.

Scope
-----
The referee scores onset localization in time, not classification. The candidate gate emits a binary
fire decision per frame, so ``Clip.onset_frames`` collapses the per-frame class labels to onset frame
positions, and the referee grades whether a fire lands within the collar of a true onset. This module
performs evaluator-side arithmetic only and charges no arm compute. It establishes no scientific claim:
the artifact assembler stamps activation_allowed=false and scientific_promotion=false, and on synthetic
fixtures the verdict can never exceed mechanics-ok.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

from .schema import COLLAR_FRAMES, FRAME_MS, Clip

REFEREE_SCHEMA = "mop-starss23-referee/v1"

# The collar as physical time and as frames on the frozen 100 ms grid. Kept as module constants so the
# sealed output records exactly which rule graded it.
COLLAR_MS = COLLAR_FRAMES * FRAME_MS  # 200 ms on each side
COLLAR_RULE = "dcase-fixed-collar-plus-minus-200ms"
MATCH_RULE = "greedy-one-to-one-nearest-first"
PR_RULE = "strict-point-wise-no-point-adjustment"


class RefereeRefusal(ValueError):
    """Raised when frames, a collar, or a clip pairing violate the sealed referee contract."""


def _require_collar(collar_frames: int) -> int:
    if isinstance(collar_frames, bool) or not isinstance(collar_frames, int) or collar_frames < 0:
        raise RefereeRefusal("collar_frames must be a nonnegative integer")
    return collar_frames


def _normalize_frames(frames: Iterable[int], label: str) -> tuple[int, ...]:
    """Validate that every frame is a nonnegative integer and return them sorted ascending.

    Frames are not deduplicated: two distinct onsets that share a frame stay distinct so the one-to-one
    matcher cannot silently merge them. Predicted fires are one per frame by construction.
    """

    normalized: list[int] = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise RefereeRefusal(f"{label} must contain only nonnegative integer frames")
        normalized.append(frame)
    normalized.sort()
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The deterministic outcome of matching predicted onset frames against ground-truth onset frames."""

    matched: tuple[tuple[int, int], ...]  # (gt_frame, pred_frame) pairs, nearest-first order of binding
    unmatched_gt: tuple[int, ...]
    unmatched_pred: tuple[int, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matched)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_gt)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_pred)


def greedy_match(
    gt_frames: Iterable[int],
    pred_frames: Iterable[int],
    collar_frames: int = COLLAR_FRAMES,
) -> MatchResult:
    """Greedy one-to-one nearest-first matching of predictions to ground truth within the collar.

    The rule is written out fully in the module docstring so an independent verifier can reproduce it.
    """

    _require_collar(collar_frames)
    gt = _normalize_frames(gt_frames, "gt_frames")
    pred = _normalize_frames(pred_frames, "pred_frames")

    candidates: list[tuple[int, int, int, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            distance = abs(p - g)
            if distance <= collar_frames:
                candidates.append((distance, g, p, gi, pi))
    candidates.sort()

    used_gt = [False] * len(gt)
    used_pred = [False] * len(pred)
    matched: list[tuple[int, int]] = []
    for _distance, g, p, gi, pi in candidates:
        if used_gt[gi] or used_pred[pi]:
            continue
        used_gt[gi] = True
        used_pred[pi] = True
        matched.append((g, p))

    unmatched_gt = tuple(g for gi, g in enumerate(gt) if not used_gt[gi])
    unmatched_pred = tuple(p for pi, p in enumerate(pred) if not used_pred[pi])
    return MatchResult(matched=tuple(matched), unmatched_gt=unmatched_gt, unmatched_pred=unmatched_pred)


def _strict_pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Strict point-wise precision, recall, and F1 from pooled counts. Undefined ratios collapse to 0."""

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


@dataclass(frozen=True, slots=True)
class OnsetScore:
    """A sealed onset-F1 score: pooled counts and the strict point-wise precision, recall, and F1."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> OnsetScore:
        for name, value in (("tp", tp), ("fp", fp), ("fn", fn)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RefereeRefusal(f"OnsetScore.{name} must be a nonnegative integer")
        precision, recall, f1 = _strict_pr(tp, fp, fn)
        return cls(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

    def payload(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def score_clip(
    gt_frames: Iterable[int],
    pred_frames: Iterable[int],
    collar_frames: int = COLLAR_FRAMES,
) -> tuple[int, int, int]:
    """Score one clip and return its pooled-ready ``(tp, fp, fn)`` counts."""

    result = greedy_match(gt_frames, pred_frames, collar_frames)
    return result.true_positives, result.false_positives, result.false_negatives


def score_arm(
    clips: Iterable[tuple[Sequence[int], Sequence[int]]],
    collar_frames: int = COLLAR_FRAMES,
) -> OnsetScore:
    """Micro-average an arm across clips.

    ``clips`` is an iterable of ``(gt_frames, pred_frames)`` pairs. Counts are summed across clips and a
    single strict point-wise precision, recall, and F1 are computed from the pooled totals.
    """

    tp = fp = fn = 0
    for gt_frames, pred_frames in clips:
        clip_tp, clip_fp, clip_fn = score_clip(gt_frames, pred_frames, collar_frames)
        tp += clip_tp
        fp += clip_fp
        fn += clip_fn
    return OnsetScore.from_counts(tp, fp, fn)


def fire_spread(
    clips: Iterable[tuple[Sequence[int], Sequence[int]]],
    collar_frames: int = COLLAR_FRAMES,
) -> dict[str, Any]:
    """Return pooled fires, adjacency, and distinct-onset score counts for one arm and seed."""

    pairs = [(list(gt), list(fires)) for gt, fires in clips]
    fire_lists = [fires for _gt, fires in pairs]
    total = adjacent = 0
    for fires in fire_lists:
        ordered = sorted(fires)
        total += len(ordered)
        for index, frame in enumerate(ordered):
            near_prev = index > 0 and frame - ordered[index - 1] <= collar_frames
            near_next = index < len(ordered) - 1 and ordered[index + 1] - frame <= collar_frames
            adjacent += near_prev or near_next
    score = score_arm(pairs, collar_frames)
    return {
        "fires": total,
        "adjacency_fraction": round(adjacent / total if total > 0 else 0.0, 12),
        "distinct_onset_tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
    }


def summarize_fire_spread(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-seed fire-spread rows without changing their order."""

    per_seed = list(rows)

    def mean(field: str) -> float:
        return math.fsum(row[field] for row in per_seed) / len(per_seed) if per_seed else 0.0

    return {
        "per_seed_fires": [row["fires"] for row in per_seed],
        "per_seed_adjacency_fraction": [row["adjacency_fraction"] for row in per_seed],
        "per_seed_distinct_onset_tp": [row["distinct_onset_tp"] for row in per_seed],
        "mean_fires": round(mean("fires"), 6),
        "mean_adjacency_fraction": round(mean("adjacency_fraction"), 12),
        "mean_distinct_onset_tp": round(mean("distinct_onset_tp"), 6),
    }


def summarize_fire_spread_blocks(
    per_seed: Iterable[dict[str, Any]],
    arm: str,
    collar_frames: int = COLLAR_FRAMES,
) -> dict[str, Any]:
    """Score and summarize one arm from producer per-seed clip blocks."""

    return summarize_fire_spread(
        fire_spread(
            ((clip["gt_onsets"], clip["fires"][arm]) for clip in block["clips"]),
            collar_frames,
        )
        for block in per_seed
    )


def onset_frames_of(clip: Clip) -> tuple[int, ...]:
    """Collapse a clip's per-frame onset labels to the onset frame positions the referee grades."""

    return clip.onset_frames


def sealed_arm_report(
    clips: Iterable[tuple[Sequence[int], Sequence[int]]],
    collar_frames: int = COLLAR_FRAMES,
) -> dict[str, Any]:
    """Score an arm and wrap it in a byte-sealed referee block.

    The block records the exact rule that graded the arm plus a canonical digest, so the independent
    verifier can confirm both the numbers and the grading protocol from the sealed artifact alone.
    """

    score = score_arm(clips, collar_frames)
    body = {
        "schema": REFEREE_SCHEMA,
        "collar_frames": collar_frames,
        "collar_ms": collar_frames * FRAME_MS,
        "collar_rule": COLLAR_RULE,
        "match_rule": MATCH_RULE,
        "pr_rule": PR_RULE,
        "score": score.payload(),
    }
    body["digest"] = canonical_sha256(body)
    return body
