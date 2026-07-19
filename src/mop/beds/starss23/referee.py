
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

from .schema import COLLAR_FRAMES, FRAME_MS, Clip

REFEREE_SCHEMA = "mop-starss23-referee/v1"

COLLAR_MS = COLLAR_FRAMES * FRAME_MS  # 200 ms on each side
COLLAR_RULE = "dcase-fixed-collar-plus-minus-200ms"
MATCH_RULE = "greedy-one-to-one-nearest-first"
PR_RULE = "strict-point-wise-no-point-adjustment"


class RefereeRefusal(ValueError):
    pass


def _require_collar(collar_frames: int) -> int:
    if isinstance(collar_frames, bool) or not isinstance(collar_frames, int) or collar_frames < 0:
        raise RefereeRefusal("collar_frames must be a nonnegative integer")
    return collar_frames


def _normalize_frames(frames: Iterable[int], label: str) -> tuple[int, ...]:

    normalized: list[int] = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise RefereeRefusal(f"{label} must contain only nonnegative integer frames")
        normalized.append(frame)
    normalized.sort()
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MatchResult:

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

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


@dataclass(frozen=True, slots=True)
class OnsetScore:

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

    result = greedy_match(gt_frames, pred_frames, collar_frames)
    return result.true_positives, result.false_positives, result.false_negatives


def score_arm(
    clips: Iterable[tuple[Sequence[int], Sequence[int]]],
    collar_frames: int = COLLAR_FRAMES,
) -> OnsetScore:

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

    return summarize_fire_spread(
        fire_spread(
            ((clip["gt_onsets"], clip["fires"][arm]) for clip in block["clips"]),
            collar_frames,
        )
        for block in per_seed
    )


def onset_frames_of(clip: Clip) -> tuple[int, ...]:

    return clip.onset_frames


def sealed_arm_report(
    clips: Iterable[tuple[Sequence[int], Sequence[int]]],
    collar_frames: int = COLLAR_FRAMES,
) -> dict[str, Any]:

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
