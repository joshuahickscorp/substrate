from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mop.evidence import canonical_sha256
from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS

from . import CLAIM_SCOPE
from .count_estimator import FLOPS_PER_REESTIMATE
from .count_gate import C_TRAIN_ANCHOR
from .count_referee import COLD_START, METRIC_RULE
from .experiments import COUNT_BED_ID
from .schema import FRAME_MS

COUNT_PREREG_SCHEMA = "mop-starss23-count-bed-prereg/v1"
STAGE = 3

PREREG_METRIC = "coasted concurrent-source count MAE (lower is better), pooled frame micro-average"
PREREG_DIRECTION = (
    "candidate < rate_matched_random (the trained gate places the same re-estimation budget better, "
    "reaching a lower count-MAE at matched re-estimation count)"
)

DEFAULT_C_TRAIN_FLOPS = C_TRAIN_ANCHOR
DEFAULT_C_REEST_FLOPS = FLOPS_PER_REESTIMATE

PREREGISTERED_SESOI_MAE = 0.02

N_PAIRED_SEEDS = 5

_FRAMES_PER_SECOND = 1000.0 / FRAME_MS  # 10 frames per second


class CountPreregRefusal(ValueError):
    pass


def _require_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CountPreregRefusal(f"{label} must be a finite number")
    if float(value) <= 0.0:
        raise CountPreregRefusal(f"{label} must be positive")
    return float(value)


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

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise CountPreregRefusal("timestamp must be a non-empty string passed by the caller")
    sesoi = _require_positive(sesoi_mae, "sesoi_mae")
    if n_seeds <= 0:
        raise CountPreregRefusal("n_seeds must be positive")

    c_train_flops = int(_require_positive(c_train_flops, "c_train_flops"))
    c_reest_flops = int(_require_positive(c_reest_flops, "c_reest_flops"))
    rho = _require_positive(operating_reestimate_fraction, "operating_reestimate_fraction")
    if not 0.0 < rho < 1.0:
        raise CountPreregRefusal("operating_reestimate_fraction must be strictly between 0 and 1")
    if n_test_frames <= 0 or n_test_clips <= 0 or n_test_changes <= 0:
        raise CountPreregRefusal("n_test_frames, n_test_clips, and n_test_changes must be positive")
    if isinstance(coast_from_zero_mae, bool) or not isinstance(coast_from_zero_mae, (int, float)):
        raise CountPreregRefusal("coast_from_zero_mae must be a real number")

    per_reestimate_saving = (1.0 - rho) * c_reest_flops
    break_even = c_train_flops / per_reestimate_saving
    break_even_hours = break_even / _FRAMES_PER_SECOND / 3600.0
    reestimate_equivalents = c_train_flops / c_reest_flops
    per_frame_granularity = 1.0 / n_test_frames
    changes_per_clip = n_test_changes / n_test_clips
    mean_run_frames = n_test_frames / n_test_changes
    one_clip_change_mass_frames = changes_per_clip * (mean_run_frames / 2.0)
    one_clip_change_mass_mae = one_clip_change_mass_frames / n_test_frames
    sesoi_in_frame_errors = sesoi * n_test_frames
    granularity_multiple = sesoi / per_frame_granularity
    rationale = (
        "Cost-benefit SESOI on coasted count-MAE. The candidate and the rate-matched-random control spend "
        "the same re-estimation count K at equal FLOPs, so the amortized training cost C_train "
        f"({c_train_flops} FLOPs, equal to {reestimate_equivalents:.0f} re-estimation equivalents at "
        f"C_reest = {c_reest_flops} FLOPs) buys nothing but the count-MAE advantage of learned re-estimation "
        "placement over free random placement. At the operating "
        f"re-estimation fraction {rho:.3f} the per-re-estimation saving against always-on is "
        f"{per_reestimate_saving:.0f} FLOPs, so the gate does not even repay C_train until "
        f"N* = {break_even:,.0f} frames (about {break_even_hours:.1f} hours of audio). Given that deployment "
        "scale, the smallest count-MAE win worth registering is one that is both measurable above the "
        "pseudoreplication floor and economically meaningful. Pooled MAE has a per-frame "
        f"granularity of {per_frame_granularity:.2e} (1 / {n_test_frames} test frames), so the chosen SESOI "
        f"of {sesoi:.2f} is {granularity_multiple:.0f}x the granularity floor (about "
        f"{sesoi_in_frame_errors:.0f} frame-errors), far above measurement noise. It also sits at about one "
        f"test clip's worth of correctly tracked changes: the test fold carries {n_test_changes} changes "
        f"over {n_test_clips} clips ({changes_per_clip:.1f} per clip) with a mean run of "
        f"{mean_run_frames:.1f} frames, so one clip's catchable change mass is about "
        f"{one_clip_change_mass_frames:.0f} frame-errors ({one_clip_change_mass_mae:.4f} pooled MAE). A win "
        "below the SESOI recovers less than about one clip of change-tracking over free random placement and "
        "does not justify carrying a trained module over the zero-training control, so it is not promotable "
        "even if the one-sided sign-flip p clears alpha."
    )
    cost_benefit = {
        "c_train_flops": c_train_flops,
        "c_reest_flops": c_reest_flops,
        "operating_reestimate_fraction": round(rho, 12),
        "per_reestimate_saving": round(per_reestimate_saving, 6),
        "break_even_frames": round(break_even, 3),
        "break_even_hours": round(break_even_hours, 4),
        "train_flops_in_reestimate_equivalents": round(reestimate_equivalents, 3),
        "n_test_frames": n_test_frames,
        "n_test_clips": n_test_clips,
        "n_test_changes": n_test_changes,
        "per_frame_granularity": round(per_frame_granularity, 9),
        "changes_per_clip": round(changes_per_clip, 6),
        "mean_run_frames": round(mean_run_frames, 6),
        "one_clip_change_mass_frames": round(one_clip_change_mass_frames, 6),
        "one_clip_change_mass_mae": round(one_clip_change_mass_mae, 9),
        "coast_from_zero_mae": round(float(coast_from_zero_mae), 9),
    }
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
            "rationale": rationale,
            "cost_benefit": cost_benefit,
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
