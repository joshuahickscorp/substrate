"""Adversarial reproduction 1 (data-split axis): preregistration of the swapped-fold SESOI.

This is a net-new, additive component. It sits beside the sealed counting bed and edits nothing under
it. It preregisters the smallest-effect-size-of-interest for the DATA-SPLIT reproduction, in which the
room-fold partition is swapped: the gate trains on the rooms the original bed scored (fold-4) and is
scored on the rooms the original bed trained and tuned on (fold-3). The two folds share no room, so the
swapped split is still genuinely room-disjoint and clip-disjoint.

The SESOI is fixed by the SAME cost-benefit rule the sealed bed used, reused by import
(``count_prereg.compute_count_cost_benefit``), evaluated on THIS reproduction's own label-only structural
facts (the fold-3 test-clip count, the fold-3 test-frame count, the fold-3 test change count, and the
fold-4 train-density-derived operating fraction). The registered number is this reproduction's own
one-test-clip catchable-change-mass on the count-MAE scale:

    SESOI = one_clip_change_mass_mae = (changes_per_clip * mean_run_frames / 2) / n_test_frames
          = 0.5 / n_test_clips   on the pooled-frame scale

so with the 24 fold-3 test clips it lands near 0.0208, on the same count scale as the original 0.02. The
prereg refuses unless the SESOI is at least 100x the per-frame measurement granularity, keeping it far
above the pseudoreplication floor.

Nothing here reads a test score. The body carries ``activation_allowed=false`` and
``scientific_promotion=false`` and a fixed timestamp that is passed in, never read from the wall clock. It
must be written before the run reads any test-split MAE.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE
from .count_prereg import (
    DEFAULT_C_REEST_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    N_PAIRED_SEEDS,
    compute_count_cost_benefit,
)
from .count_referee import COLD_START, METRIC_RULE
from .experiments import COUNT_BED_ID

REPRO_PREREG_SCHEMA = "mop-starss23-count-repro-data-split-prereg/v1"
REPRO_AXIS = "data_split"
STAGE = 3

# The reproduction registers its own self-derived SESOI, not a hardcoded round number, so the check that
# it be at least this multiple of the per-frame granularity is what keeps it above the pseudoreplication
# floor. The original clears the same floor at 451x.
MIN_GRANULARITY_MULTIPLE = 100.0

PREREG_METRIC = (
    "coasted concurrent-source count MAE (lower is better), pooled frame micro-average, swapped fold split"
)
PREREG_DIRECTION = (
    "candidate < rate_matched_random on the SWAPPED split (train on the original test rooms, score on the "
    "original train rooms): the trained gate places the same re-estimation budget better, reaching a lower "
    "count-MAE at matched re-estimation count"
)


class ReproPreregRefusal(ValueError):
    """Raised when a data-split reproduction preregistration input is malformed."""


def _swapped_split_rationale(cb: Any, sesoi: float, granularity_multiple: float) -> str:
    return (
        "Data-split reproduction SESOI on coasted count-MAE, reusing the sealed bed's cost-benefit rule on "
        "the SWAPPED fold split. The candidate and the rate-matched-random control spend the same "
        "re-estimation count K at equal FLOPs, so the amortized training cost C_train "
        f"({cb.c_train_flops} FLOPs, equal to {cb.train_flops_in_reestimate_equivalents:.0f} re-estimation "
        f"equivalents at C_reest = {cb.c_reest_flops} FLOPs) buys nothing but the count-MAE advantage of "
        "learned re-estimation placement over free random placement. The registered SESOI is this "
        "reproduction's own one-test-clip catchable-change mass, derived in code from the fold-3 test "
        f"labels: {cb.n_test_changes} changes over {cb.n_test_clips} clips ({cb.changes_per_clip:.1f} per "
        f"clip) with a mean run of {cb.mean_run_frames:.1f} frames give one clip's catchable change mass of "
        f"about {cb.one_clip_change_mass_frames:.0f} frame-errors ({sesoi:.6f} pooled MAE), which reduces to "
        f"0.5 / {cb.n_test_clips} test clips. Pooled MAE has a per-frame granularity of "
        f"{cb.per_frame_granularity:.2e} (1 / {cb.n_test_frames} test frames), so this SESOI is "
        f"{granularity_multiple:.0f}x the granularity floor, far above measurement noise. A win below it "
        "recovers less than about one clip of change-tracking over free random placement and is not "
        "promotable even if the one-sided sign-flip p clears alpha."
    )


def build_data_split_prereg(
    *,
    timestamp: str,
    operating_reestimate_fraction: float,
    n_test_clips: int,
    n_test_changes: int,
    n_test_frames: int,
    train_change_density: float,
    coast_from_zero_mae: float,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_reest_flops: int = DEFAULT_C_REEST_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
) -> dict[str, Any]:
    """Assemble the self-sealed data-split reproduction preregistration.

    The SESOI is computed from the fold-3 test labels by the reused cost-benefit rule and refused unless it
    clears the granularity floor. The timestamp is passed by the caller, never read from a clock. No test
    score is read.
    """

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ReproPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise ReproPreregRefusal("n_seeds must be positive")

    cb = compute_count_cost_benefit(
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
        n_test_frames=n_test_frames,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        coast_from_zero_mae=coast_from_zero_mae,
    )

    # The reproduction's own SESOI: the one-test-clip catchable-change mass, self-derived from the fold-3
    # test labels. This is 0.5 / n_test_clips on the pooled-frame scale and is set BEFORE any test MAE.
    sesoi_mae = float(cb.one_clip_change_mass_mae)
    if not (sesoi_mae > 0.0):
        raise ReproPreregRefusal("derived SESOI must be positive")
    granularity_multiple = sesoi_mae / cb.per_frame_granularity
    if granularity_multiple < MIN_GRANULARITY_MULTIPLE:
        raise ReproPreregRefusal(
            f"derived SESOI {sesoi_mae} is only {granularity_multiple:.1f}x the per-frame granularity, "
            f"below the {MIN_GRANULARITY_MULTIPLE:.0f}x pseudoreplication floor"
        )

    permutations = 2**n_seeds
    body: dict[str, Any] = {
        "schema": REPRO_PREREG_SCHEMA,
        "reproduction_axis": REPRO_AXIS,
        "of_bed": COUNT_BED_ID,
        "stage": STAGE,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "spurious_win_targeted": (
            "that the sealed bed's win is specific to training on fold-3 rooms and scoring on fold-4, a "
            "room-idiosyncratic coincidence rather than a generic 'place re-estimations near count changes "
            "beats uniform placement'"
        ),
        "question": (
            "does the trained gate still reach lower coasted count-MAE than rate-matched-random when the "
            "room-fold partition is swapped (train on the original test rooms, score on the original train "
            "rooms), at the same re-estimation budget"
        ),
        "metric": PREREG_METRIC,
        "metric_rule": METRIC_RULE,
        "cold_start": COLD_START,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_mae": round(sesoi_mae, 12),
            "provisional": False,
            "selection_method": (
                "reused cost-benefit rule; registered value is the reproduction's own one-test-clip "
                "catchable-change mass on the fold-3 test labels (0.5 / n_test_clips)"
            ),
            "granularity_multiple": round(float(granularity_multiple), 6),
            "min_granularity_multiple": MIN_GRANULARITY_MULTIPLE,
            "rationale": _swapped_split_rationale(cb, sesoi_mae, granularity_multiple),
            "cost_benefit": cb.payload(),
            "train_change_density": round(float(train_change_density), 12),
        },
        "operating_point_rule": (
            "the swept re-estimation budget whose fraction is closest to the SWAPPED train-set (fold-4) "
            "count-change density; a fixed rule set before scoring, using only train labels, never a val or "
            "test MAE argmax"
        ),
        "sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "seed_family": "disjoint from the original (0..4): data-split uses (10..14)",
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
        },
        "survive_criterion": (
            "candidate mean count-MAE strictly below rate_matched_random AND mean paired delta "
            ">= registered SESOI AND one-sided sign-flip p <= 1/32; any tie is a null"
        ),
        "promotion_bar": (
            "a single reproduction can never promote; independent_scientific_confirmation is set only by the "
            "separately authored verifier and stays false until at least three bias-independent "
            "reproductions are on record"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body

DEFAULT_REPRO_PREREG_PATH = Path("proof/STARSS23_COUNTING_REPRO_data_split.prereg.json")
