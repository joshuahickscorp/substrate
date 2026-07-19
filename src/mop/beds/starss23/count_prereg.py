"""Preregistration of the STARSS23 concurrent-source-counting bed SESOI and analysis plan.

This is a net-new, additive component. It fixes the smallest-effect-size-of-interest (SESOI) on count-MAE
by explicit cost-benefit reasoning and freezes the whole analysis plan into a self-sealed artifact
``proof/STARSS23_COUNTING_BED.prereg.json`` that MUST be written before the run reads any test-split score.

Nothing here reads a test score. The cost-benefit derivation uses only fixed compute anchors (the gate's
amortized C_train and the per-re-estimation estimator cost C_reest) and structural facts of the corpus
known from the labels before any arm is scored (the test-clip count, the test-frame count, the pooled
test count-change count, and the never-update drift scale). The sealed body carries
``activation_allowed=false`` and ``scientific_promotion=false`` and a fixed timestamp that is passed in,
never read from the wall clock.

The multiplicity wall is the exact sign-flip permutation floor: with five paired seeds there are 2^5 = 32
sign assignments, so the minimum one-sided p is 1/32 and two-sided 0.05 is unreachable. This is preregistered
in code below.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE
from .count_estimator import FLOPS_PER_REESTIMATE
from .count_referee import COLD_START, METRIC_RULE
from .experiments import COUNT_BED_ID
from .gate import C_TRAIN_ANCHOR
from .schema import FRAME_MS

COUNT_PREREG_SCHEMA = "mop-starss23-count-bed-prereg/v1"
STAGE = 3

PREREG_METRIC = "coasted concurrent-source count MAE (lower is better), pooled frame micro-average"
PREREG_DIRECTION = (
    "candidate < rate_matched_random (the trained gate places the same re-estimation budget better, "
    "reaching a lower count-MAE at matched re-estimation count)"
)

# The amortized training cost anchor, reused unchanged from gate.py: C_train = 8 * 54000 * 3 * 6385 ~ 8.27e9.
DEFAULT_C_TRAIN_FLOPS = C_TRAIN_ANCHOR
# The per-re-estimation estimator cost anchor, reused from count_estimator.py: 4x4 covariance plus eigen ~ 8e4.
DEFAULT_C_REEST_FLOPS = FLOPS_PER_REESTIMATE

# The preregistered SESOI on count-MAE, fixed here by the cost-benefit derivation. Counts per frame.
PREREGISTERED_SESOI_MAE = 0.02

N_PAIRED_SEEDS = 5

# Frames per second on the 100 ms label grid, for the break-even-hours readout.
_FRAMES_PER_SECOND = 1000.0 / FRAME_MS  # 10 frames per second


class CountPreregRefusal(ValueError):
    """Raised when a count preregistration input is malformed."""


def _require_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CountPreregRefusal(f"{label} must be a finite number")
    if float(value) <= 0.0:
        raise CountPreregRefusal(f"{label} must be positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class CountCostBenefit:
    """The cost-benefit inputs and derived break-even that justify the SESOI on the count-MAE scale."""

    c_train_flops: int
    c_reest_flops: int
    operating_reestimate_fraction: float
    per_reestimate_saving: float
    break_even_frames: float
    break_even_hours: float
    train_flops_in_reestimate_equivalents: float
    n_test_frames: int
    n_test_clips: int
    n_test_changes: int
    per_frame_granularity: float
    changes_per_clip: float
    mean_run_frames: float
    one_clip_change_mass_frames: float
    one_clip_change_mass_mae: float
    coast_from_zero_mae: float

    def payload(self) -> dict[str, Any]:
        return {
            "c_train_flops": self.c_train_flops,
            "c_reest_flops": self.c_reest_flops,
            "operating_reestimate_fraction": round(self.operating_reestimate_fraction, 12),
            "per_reestimate_saving": round(self.per_reestimate_saving, 6),
            "break_even_frames": round(self.break_even_frames, 3),
            "break_even_hours": round(self.break_even_hours, 4),
            "train_flops_in_reestimate_equivalents": round(self.train_flops_in_reestimate_equivalents, 3),
            "n_test_frames": self.n_test_frames,
            "n_test_clips": self.n_test_clips,
            "n_test_changes": self.n_test_changes,
            "per_frame_granularity": round(self.per_frame_granularity, 9),
            "changes_per_clip": round(self.changes_per_clip, 6),
            "mean_run_frames": round(self.mean_run_frames, 6),
            "one_clip_change_mass_frames": round(self.one_clip_change_mass_frames, 6),
            "one_clip_change_mass_mae": round(self.one_clip_change_mass_mae, 9),
            "coast_from_zero_mae": round(self.coast_from_zero_mae, 9),
        }


def compute_count_cost_benefit(
    *,
    c_train_flops: int,
    c_reest_flops: int,
    operating_reestimate_fraction: float,
    n_test_frames: int,
    n_test_clips: int,
    n_test_changes: int,
    coast_from_zero_mae: float,
) -> CountCostBenefit:
    """Derive the break-even and the count-MAE granularity that bound the SESOI. Reads no test score."""

    c_train_flops = int(_require_positive(c_train_flops, "c_train_flops"))
    c_reest_flops = int(_require_positive(c_reest_flops, "c_reest_flops"))
    rho = _require_positive(operating_reestimate_fraction, "operating_reestimate_fraction")
    if not 0.0 < rho < 1.0:
        raise CountPreregRefusal("operating_reestimate_fraction must be strictly between 0 and 1")
    if n_test_frames <= 0 or n_test_clips <= 0 or n_test_changes <= 0:
        raise CountPreregRefusal("n_test_frames, n_test_clips, and n_test_changes must be positive")
    if isinstance(coast_from_zero_mae, bool) or not isinstance(coast_from_zero_mae, (int, float)):
        raise CountPreregRefusal("coast_from_zero_mae must be a real number")

    # Per-re-estimation cost saved against always-on: always-on re-estimates every frame, the gate only a
    # fraction rho, so each re-estimation the gate declines saves (1 - rho) * C_reest downstream FLOPs.
    per_reestimate_saving = (1.0 - rho) * c_reest_flops
    break_even = c_train_flops / per_reestimate_saving
    break_even_hours = break_even / _FRAMES_PER_SECOND / 3600.0
    reestimate_equivalents = c_train_flops / c_reest_flops

    # Measurement granularity: one frame of pooled absolute error moves pooled MAE by 1 / n_test_frames.
    per_frame_granularity = 1.0 / n_test_frames
    # One test clip's worth of catchable count changes. A missed +/- 1 change coasted for about half of its
    # mean run costs about (mean_run / 2) unit frame-errors; a clip carries about changes_per_clip changes.
    changes_per_clip = n_test_changes / n_test_clips
    mean_run_frames = n_test_frames / n_test_changes
    one_clip_change_mass_frames = changes_per_clip * (mean_run_frames / 2.0)
    one_clip_change_mass_mae = one_clip_change_mass_frames / n_test_frames

    return CountCostBenefit(
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=rho,
        per_reestimate_saving=per_reestimate_saving,
        break_even_frames=break_even,
        break_even_hours=break_even_hours,
        train_flops_in_reestimate_equivalents=reestimate_equivalents,
        n_test_frames=n_test_frames,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        per_frame_granularity=per_frame_granularity,
        changes_per_clip=changes_per_clip,
        mean_run_frames=mean_run_frames,
        one_clip_change_mass_frames=one_clip_change_mass_frames,
        one_clip_change_mass_mae=one_clip_change_mass_mae,
        coast_from_zero_mae=float(coast_from_zero_mae),
    )


def _sesoi_rationale(cb: CountCostBenefit, sesoi: float) -> str:
    sesoi_in_frame_errors = sesoi * cb.n_test_frames
    granularity_multiple = sesoi / cb.per_frame_granularity
    return (
        "Cost-benefit SESOI on coasted count-MAE. The candidate and the rate-matched-random control spend "
        "the same re-estimation count K at equal FLOPs, so the amortized training cost C_train "
        f"({cb.c_train_flops} FLOPs, equal to {cb.train_flops_in_reestimate_equivalents:.0f} "
        f"re-estimation equivalents at C_reest = {cb.c_reest_flops} FLOPs) buys nothing but the count-MAE "
        "advantage of learned re-estimation placement over free random placement. At the operating "
        f"re-estimation fraction {cb.operating_reestimate_fraction:.3f} the per-re-estimation saving against "
        f"always-on is {cb.per_reestimate_saving:.0f} FLOPs, so the gate does not even repay C_train until "
        f"N* = {cb.break_even_frames:,.0f} frames (about {cb.break_even_hours:.1f} hours of audio). Given "
        "that deployment scale, the smallest count-MAE win worth registering is one that is both measurable "
        "above the pseudoreplication floor and economically meaningful. Pooled MAE has a per-frame "
        f"granularity of {cb.per_frame_granularity:.2e} (1 / {cb.n_test_frames} test frames), so the chosen "
        f"SESOI of {sesoi:.2f} is {granularity_multiple:.0f}x the granularity floor (about "
        f"{sesoi_in_frame_errors:.0f} frame-errors), far above measurement noise. It also sits at about one "
        f"test clip's worth of correctly tracked changes: the test fold carries {cb.n_test_changes} changes "
        f"over {cb.n_test_clips} clips ({cb.changes_per_clip:.1f} per clip) with a mean run of "
        f"{cb.mean_run_frames:.1f} frames, so one clip's catchable change mass is about "
        f"{cb.one_clip_change_mass_frames:.0f} frame-errors ({cb.one_clip_change_mass_mae:.4f} pooled MAE). "
        "A win below the SESOI recovers less than about one clip of change-tracking over free random "
        "placement and does not justify carrying a trained module over the zero-training control, so it is "
        "not promotable even if the one-sided sign-flip p clears alpha."
    )


def build_count_prereg(
    *,
    timestamp: str,
    operating_reestimate_fraction: float,
    n_test_clips: int,
    n_test_changes: int,
    n_test_frames: int,
    train_change_density: float,
    coast_from_zero_mae: float,
    sesoi_mae: float = PREREGISTERED_SESOI_MAE,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_reest_flops: int = DEFAULT_C_REEST_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
) -> dict[str, Any]:
    """Assemble the self-sealed count preregistration body. The timestamp is passed, never read from a clock."""

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise CountPreregRefusal("timestamp must be a non-empty string passed by the caller")
    sesoi = _require_positive(sesoi_mae, "sesoi_mae")
    if n_seeds <= 0:
        raise CountPreregRefusal("n_seeds must be positive")

    cb = compute_count_cost_benefit(
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
        n_test_frames=n_test_frames,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        coast_from_zero_mae=coast_from_zero_mae,
    )
    permutations = 2**n_seeds

    body: dict[str, Any] = {
        "schema": COUNT_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": COUNT_BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "question": (
            "concurrent-source COUNTING value-of-computation: does a trained gate reach lower coasted "
            "count-MAE at the same re-estimation budget as a rate-matched-random gate"
        ),
        "metric": PREREG_METRIC,
        "metric_rule": METRIC_RULE,
        "cold_start": COLD_START,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_mae": round(sesoi, 12),
            "provisional": False,
            "selection_method": "cost-benefit (deployment break-even plus one-test-clip change mass)",
            "rationale": _sesoi_rationale(cb, sesoi),
            "cost_benefit": cb.payload(),
            "train_change_density": round(float(train_change_density), 12),
        },
        "operating_point_rule": (
            "the swept re-estimation budget whose fraction is closest to the train-set count-change "
            "density; a fixed rule set before scoring, using only train labels, never a val or test MAE "
            "argmax"
        ),
        "sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "statistic": "mean of paired per-seed count-MAE deltas (rate_matched_random minus candidate)",
            "min_one_sided_p": round(1.0 / permutations, 12),
            "two_sided_floor": round(2.0 / permutations, 12),
            "alpha": 0.05,
            "two_sided_alpha_reachable": (2.0 / permutations) <= 0.05,
            "phipson_smyth_applied": False,
        },
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_frames": int(n_test_frames),
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
            "rationale": (
                "with about "
                f"{n_test_clips} test clips the clip is the experimental unit and frames are correlated "
                "sub-samples; a frame or clip bootstrap is refused and the claim verb is bounded to "
                "'consistent with', never 'demonstrates' or 'significant'"
            ),
        },
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears alpha "
            "AND at least three bias-independent reproductions triangulate the same direction; a single run "
            "can never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


DEFAULT_COUNT_PREREG_PATH = Path("proof/STARSS23_COUNTING_BED.prereg.json")


def write_count_prereg(body: dict[str, Any], out_path: str | Path = DEFAULT_COUNT_PREREG_PATH) -> Path:
    """Write the self-sealed preregistration as canonical JSON bytes so its on-disk digest is stable."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path
