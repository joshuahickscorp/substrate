"""Scoring-unit adversarial reproduction, component B: the clip-macro SESOI preregistration.

Net-new and additive. It fixes the smallest-effect-size-of-interest for the clip-macro reproduction on the
clip-macro count-MAE scale and freezes the whole analysis plan into a self-sealed artifact
``proof/STARSS23_COUNTING_REPRO_scoring_unit.prereg.json`` that MUST be written before the run reads any
test-split score.

The SESOI is derived by the SAME cost-benefit rule the sealed bed uses, reused BY IMPORT: it calls
``count_prereg.compute_count_cost_benefit`` on the reproduction's own label-only structural facts (its test
clip count, test frame count, pooled test change count, and its train-density operating fraction, with the
unchanged C_train and C_reest anchors). That rule returns ``one_clip_change_mass_mae``, one test clip's
worth of catchable count changes expressed on the count-MAE scale, which is the registered SESOI.

Because this reproduction re-scores with the clip as the experimental unit, the same quantity is ALSO
restated directly on the clip-macro scale: each change-bearing clip's own catchable change mass is
``(changes_c * mean_run_c / 2) / n_frames_c`` per-clip MAE, and because each clip enters the macro mean with
weight ``1 / n_clips`` the macro-MAE gap of one clip's worth of change tracking is the clip mean of that
per-clip quantity divided by ``n_clips``. That macro restatement coincides EXACTLY with the reused pooled
rule at ``0.5 / n_clips``, so the registered number is identical on both derivations; sealing both makes the
coincidence auditable. The registered SESOI is the exact computed value, not a hand-rounded number.

The prereg refuses unless the registered SESOI clears 100 times the macro measurement granularity
``1 / (n_clips * min_clip_frames)`` (one frame of absolute error in the shortest clip, carried into the
equal-weight macro mean), keeping it far above the clip-macro pseudoreplication floor.

Nothing here reads a test score. ``activation_allowed`` and ``scientific_promotion`` are hardcoded false and
the timestamp is passed by the caller, never read from the wall clock. House style: no em dashes and no en
dashes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE
from .count_prereg import (
    DEFAULT_C_REEST_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    N_PAIRED_SEEDS,
    compute_count_cost_benefit,
)
from .count_referee import COLD_START, METRIC_RULE
from .count_repro_scoring_unit_referee import MACRO_METRIC_RULE, SCORING_UNIT
from .stats import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS

COUNT_REPRO_SCORING_UNIT_PREREG_SCHEMA = "mop-starss23-count-repro-scoring-unit-prereg/v1"
STAGE = 3
REPRO_AXIS = "scoring_unit"

# The macro SESOI must clear at least this multiple of the clip-macro measurement granularity.
MIN_GRANULARITY_MULTIPLE = 100.0

PREREG_METRIC = (
    "coasted concurrent-source count MAE re-scored with the clip as the experimental unit (clip-macro "
    "mean of per-clip MAE, lower is better)"
)
PREREG_DIRECTION = (
    "candidate < rate_matched_random on the clip-macro count-MAE (the trained gate places the same "
    "re-estimation budget better, reaching a lower per-clip-averaged count-MAE at matched re-estimation "
    "count), corroborated by the candidate being lower on the clip average in the clip-clustered readout"
)


class CountReproScoringUnitPreregRefusal(ValueError):
    """Raised when a clip-macro preregistration input is malformed or the SESOI floor is not cleared."""


@dataclass(frozen=True, slots=True)
class ClipLabelFact:
    """One test clip's label-only structural facts: its frame count and its count-change count."""

    clip_id: str
    n_frames: int
    n_changes: int


def _macro_restatement(facts: Sequence[ClipLabelFact], n_clips: int) -> tuple[float, list[dict[str, Any]]]:
    """Restate one clip's worth of catchable change mass directly on the clip-macro MAE scale.

    For a change-bearing clip the per-clip catchable change mass is
    ``(changes_c * mean_run_c / 2) / n_frames_c`` with ``mean_run_c = n_frames_c / changes_c``, which is
    exactly ``0.5`` of that clip's own frame scale; a clip with no changes contributes zero. Because each
    clip enters the equal-weight macro mean with weight ``1 / n_clips``, one clip's worth of change tracking
    moves the macro mean by the clip mean of that per-clip quantity divided by ``n_clips``.
    """

    brackets: list[dict[str, Any]] = []
    total = 0.0
    for fact in facts:
        if fact.n_changes <= 0:
            per_clip_mass = 0.0
        else:
            mean_run_c = fact.n_frames / fact.n_changes
            per_clip_mass = (fact.n_changes * (mean_run_c / 2.0)) / fact.n_frames
        total += per_clip_mass
        brackets.append(
            {
                "clip_id": fact.clip_id,
                "n_frames": fact.n_frames,
                "n_changes": fact.n_changes,
                "per_clip_change_mass_mae": round(per_clip_mass, 12),
            }
        )
    macro_restatement = (total / n_clips) / n_clips
    return macro_restatement, brackets


def build_count_repro_scoring_unit_prereg(
    *,
    timestamp: str,
    operating_reestimate_fraction: float,
    test_clip_facts: Sequence[ClipLabelFact],
    train_change_density: float,
    coast_from_zero_mae: float,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_reest_flops: int = DEFAULT_C_REEST_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
) -> dict[str, Any]:
    """Assemble the self-sealed clip-macro preregistration body. Reads no test score.

    The registered SESOI on the clip-macro count-MAE scale is the reused cost-benefit rule's
    ``one_clip_change_mass_mae`` computed from label-only facts, cross-derived by the macro restatement and
    refused unless it clears ``MIN_GRANULARITY_MULTIPLE`` times the clip-macro measurement granularity.
    """

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise CountReproScoringUnitPreregRefusal("timestamp must be a non-empty string passed by the caller")
    facts = tuple(test_clip_facts)
    if not facts:
        raise CountReproScoringUnitPreregRefusal("at least one test clip fact is required")
    for fact in facts:
        if fact.n_frames <= 0:
            raise CountReproScoringUnitPreregRefusal(f"clip {fact.clip_id!r} must have a positive frame count")
        if fact.n_changes < 0:
            raise CountReproScoringUnitPreregRefusal(f"clip {fact.clip_id!r} change count must be nonnegative")
    if n_seeds <= 0:
        raise CountReproScoringUnitPreregRefusal("n_seeds must be positive")

    n_test_clips = len(facts)
    n_test_frames = int(sum(fact.n_frames for fact in facts))
    n_test_changes = int(sum(fact.n_changes for fact in facts))
    if n_test_changes <= 0:
        raise CountReproScoringUnitPreregRefusal("the test split carries no count changes to track")
    min_clip_frames = int(min(fact.n_frames for fact in facts))

    # The registered SESOI: the reused cost-benefit rule on this reproduction's own label-only facts.
    cb = compute_count_cost_benefit(
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
        n_test_frames=n_test_frames,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        coast_from_zero_mae=coast_from_zero_mae,
    )
    sesoi_macro = float(cb.one_clip_change_mass_mae)

    macro_restatement, brackets = _macro_restatement(facts, n_test_clips)

    # The clip-macro measurement granularity: one frame of absolute error in the shortest clip moves that
    # clip's MAE by 1 / min_clip_frames and the equal-weight macro mean by 1 / (n_clips * min_clip_frames).
    macro_granularity = 1.0 / (n_test_clips * min_clip_frames)
    granularity_floor = MIN_GRANULARITY_MULTIPLE * macro_granularity
    if not math.isfinite(sesoi_macro) or sesoi_macro <= 0.0:
        raise CountReproScoringUnitPreregRefusal("the registered clip-macro SESOI must be positive and finite")
    if sesoi_macro < granularity_floor:
        raise CountReproScoringUnitPreregRefusal(
            f"registered clip-macro SESOI {sesoi_macro} is below {MIN_GRANULARITY_MULTIPLE:.0f} times the "
            f"clip-macro granularity {macro_granularity}; it would sit at the pseudoreplication floor"
        )

    permutations = 2**n_seeds
    clip_permutations = 2**n_test_clips
    rationale = (
        "Cost-benefit SESOI on the clip-macro count-MAE. The candidate and the rate-matched-random control "
        "spend the same re-estimation count at equal FLOPs, so the amortized training cost C_train "
        f"({cb.c_train_flops} FLOPs, {cb.train_flops_in_reestimate_equivalents:.0f} re-estimation "
        f"equivalents at C_reest = {cb.c_reest_flops} FLOPs) buys only the count-MAE advantage of learned "
        "re-estimation placement over free random placement. The smallest clip-macro count-MAE win worth "
        "registering is one test clip's worth of correctly tracked changes: the reused cost-benefit rule "
        f"puts that at {sesoi_macro:.6f} count-MAE. Because the clip is the experimental unit here, the same "
        f"quantity restated directly on the clip-macro scale is {macro_restatement:.6f}, which coincides "
        "exactly with the reused pooled rule at 0.5 / n_clips because each change-bearing clip contributes "
        "half of its own frame scale and enters the equal-weight macro mean with weight 1 / n_clips. The "
        f"clip-macro measurement granularity is {macro_granularity:.3e} (one frame of error in the shortest "
        f"clip of {min_clip_frames} frames, carried into the {n_test_clips}-clip macro mean), so the SESOI "
        f"sits at {sesoi_macro / macro_granularity:.0f} times that floor, far above measurement noise. A win "
        "below the SESOI recovers less than about one clip of change tracking over free random placement and "
        "is not promotable even if the one-sided sign-flip p clears alpha."
    )

    body: dict[str, Any] = {
        "schema": COUNT_REPRO_SCORING_UNIT_PREREG_SCHEMA,
        "stage": STAGE,
        "reproduction_axis": REPRO_AXIS,
        "reproduces": "starss23_escs_source_counting",
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "question": (
            "does the trained gate's lower coasted count-MAE at a matched re-estimation budget survive when "
            "the pooled frame micro-average is replaced by the clip as the experimental unit (clip-macro "
            "mean), so long-clip pseudoreplication cannot inflate it"
        ),
        "metric": PREREG_METRIC,
        "pooled_metric_rule_reference": METRIC_RULE,
        "macro_metric_rule": MACRO_METRIC_RULE,
        "scoring_unit": SCORING_UNIT,
        "cold_start": COLD_START,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_mae": round(sesoi_macro, 12),
            "scale": "clip-macro count-MAE",
            "provisional": False,
            "selection_method": (
                "reused cost-benefit rule (count_prereg.compute_count_cost_benefit) on this reproduction's "
                "own label-only facts, cross-derived by the clip-macro restatement"
            ),
            "reused_one_clip_change_mass_mae": round(sesoi_macro, 12),
            "macro_scale_restatement": round(macro_restatement, 12),
            "restatement_matches_reused_rule": abs(macro_restatement - sesoi_macro) <= 1e-12,
            "macro_granularity": round(macro_granularity, 12),
            "granularity_multiple_required": MIN_GRANULARITY_MULTIPLE,
            "granularity_multiple": round(sesoi_macro / macro_granularity, 6),
            "granularity_floor": round(granularity_floor, 12),
            "clears_granularity_floor": sesoi_macro >= granularity_floor,
            "rationale": rationale,
            "cost_benefit": cb.payload(),
            "per_clip_change_mass": brackets,
            "train_change_density": round(float(train_change_density), 12),
        },
        "operating_point_rule": (
            "the swept re-estimation budget whose fraction is closest to the train-set count-change density; "
            "a fixed rule set before scoring, using only train labels, never a val or test MAE argmax"
        ),
        "primary_sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail, on the five-seed clip-macro deltas",
            "unit": "paired seed",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "statistic": "mean of paired per-seed clip-macro count-MAE deltas (rate_matched_random minus candidate)",
            "min_one_sided_p": round(1.0 / permutations, 12),
            "two_sided_floor": round(2.0 / permutations, 12),
            "alpha": 0.05,
            "two_sided_alpha_reachable": (2.0 / permutations) <= 0.05,
            "phipson_smyth_applied": False,
        },
        "corroborating_clip_cluster_plan": {
            "test": "exact sign-flip permutation over clips (one sign per clip), one-sided, upper tail",
            "unit": "test clip",
            "n_clips": n_test_clips,
            "n_permutations": clip_permutations,
            "statistic": (
                "sum over clips of per-clip paired deltas d_c = macro MAE_clip(rate_matched_random) minus "
                "macro MAE_clip(candidate), each per-clip MAE averaged over the paired seeds at the "
                "operating point"
            ),
            "min_one_sided_p": round(1.0 / clip_permutations, 15),
            "required_to_agree_in_direction": True,
            "direction_agreement_rule": "mean per-clip delta strictly positive (candidate lower on the clip average)",
        },
        "survive_rule": (
            "SURVIVE only if the candidate clip-macro mean count-MAE is strictly below the "
            "rate-matched-random clip-macro mean AND the mean five-seed clip-macro delta exceeds the "
            "registered SESOI AND the one-sided five-seed sign-flip p clears alpha AND the clip-clustered "
            "readout agrees in direction; any tie or reversal is a null"
        ),
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_frames": n_test_frames,
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
            "rationale": (
                f"with {n_test_clips} test clips the clip is the experimental unit and frames are correlated "
                "sub-samples; this reproduction scores with the clip as the unit and additionally runs an "
                "exact clip-clustered permutation, and the claim verb is bounded to 'consistent with'"
            ),
        },
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears alpha "
            "AND at least three bias-independent reproductions triangulate the same direction; a single run "
            "can never promote, and the independent verifier alone sets independent_scientific_confirmation"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


DEFAULT_COUNT_REPRO_SCORING_UNIT_PREREG_PATH = Path(
    "proof/STARSS23_COUNTING_REPRO_scoring_unit.prereg.json"
)


def write_count_repro_scoring_unit_prereg(
    body: dict[str, Any], out_path: str | Path = DEFAULT_COUNT_REPRO_SCORING_UNIT_PREREG_PATH
) -> Path:
    """Write the self-sealed preregistration as canonical JSON bytes so its on-disk digest is stable."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path
