
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE
from .count_prereg import DEFAULT_C_TRAIN_FLOPS, compute_count_cost_benefit
from .count_referee import COLD_START, METRIC_RULE
from .count_repro_featurizer_estimator_estimator import (
    COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
    ESTIMATOR_RULE,
)
from .count_repro_featurizer_estimator_estimator import (
    FLOPS_PER_REESTIMATE as REPRO_C_REEST_FLOPS,
)
from .count_repro_featurizer_estimator_featurizer import COUNT_REPRO_FE_FEATURIZER_SCHEMA
from .experiments import COUNT_BED_ID
from .schema import FRAME_MS

COUNT_REPRO_FE_PREREG_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-prereg/v1"
REPRO_AXIS = "featurizer_estimator"
STAGE = 3

REPRO_SEEDS: tuple[int, ...] = (20, 21, 22, 23, 24)

DEFAULT_C_REEST_FLOPS = REPRO_C_REEST_FLOPS

N_PAIRED_SEEDS = 5
_FRAMES_PER_SECOND = 1000.0 / FRAME_MS  # 10 frames per second on the 100 ms grid
MIN_GRANULARITY_MULTIPLE = 100.0


class CountReproPreregRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReproSesoi:

    sesoi_mae: float
    one_clip_change_mass_mae: float
    per_frame_granularity: float
    granularity_multiple: float
    n_test_clips: int
    n_test_changes: int
    n_test_frames: int

    def payload(self) -> dict[str, Any]:
        return {
            "sesoi_mae": round(self.sesoi_mae, 12),
            "one_clip_change_mass_mae": round(self.one_clip_change_mass_mae, 12),
            "per_frame_granularity": round(self.per_frame_granularity, 12),
            "granularity_multiple": round(self.granularity_multiple, 6),
            "n_test_clips": self.n_test_clips,
            "n_test_changes": self.n_test_changes,
            "n_test_frames": self.n_test_frames,
        }


def compute_repro_sesoi(
    *,
    operating_reestimate_fraction: float,
    n_test_clips: int,
    n_test_changes: int,
    n_test_frames: int,
    coast_from_zero_mae: float,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_reest_flops: int = DEFAULT_C_REEST_FLOPS,
) -> tuple[ReproSesoi, Any]:

    cb = compute_count_cost_benefit(
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
        n_test_frames=n_test_frames,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        coast_from_zero_mae=coast_from_zero_mae,
    )
    sesoi = float(cb.one_clip_change_mass_mae)
    granularity_multiple = sesoi / cb.per_frame_granularity
    if granularity_multiple < MIN_GRANULARITY_MULTIPLE:
        raise CountReproPreregRefusal(
            f"SESOI {sesoi} is only {granularity_multiple:.1f}x the per-frame granularity; the reproduction "
            f"requires at least {MIN_GRANULARITY_MULTIPLE:.0f}x above the pseudoreplication floor"
        )
    return (
        ReproSesoi(
            sesoi_mae=sesoi,
            one_clip_change_mass_mae=float(cb.one_clip_change_mass_mae),
            per_frame_granularity=float(cb.per_frame_granularity),
            granularity_multiple=granularity_multiple,
            n_test_clips=n_test_clips,
            n_test_changes=n_test_changes,
            n_test_frames=n_test_frames,
        ),
        cb,
    )


def _sesoi_rationale(sesoi: ReproSesoi) -> str:
    return (
        "Cost-benefit SESOI on coasted count-MAE for the featurizer_estimator reproduction, recomputed in "
        "code from label-only facts by the same rule the sealed bed used and sealed before any test score. "
        "The candidate and the rate-matched-random control spend the same re-estimation count K at equal "
        "FLOPs, so a trained module is only worth carrying if it recovers at least about one test clip's "
        f"worth of correctly tracked count changes over free random placement. The test fold carries "
        f"{sesoi.n_test_changes} changes over {sesoi.n_test_clips} clips, so one clip's catchable change "
        f"mass on the pooled count-MAE scale is {sesoi.one_clip_change_mass_mae:.6f} (which equals "
        f"0.5 / {sesoi.n_test_clips} test clips). That is {sesoi.granularity_multiple:.0f}x the per-frame "
        f"measurement granularity of {sesoi.per_frame_granularity:.2e}, far above the pseudoreplication "
        "floor. The estimator swap does not move this number because the SESOI is a property of the count "
        "labels, not of the estimator track. A win below the SESOI recovers less than about one clip of "
        "change tracking over random placement and is not promotable even if the one-sided sign-flip p "
        "clears alpha."
    )


def build_repro_prereg(
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

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise CountReproPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise CountReproPreregRefusal("n_seeds must be positive")

    sesoi, cb = compute_repro_sesoi(
        operating_reestimate_fraction=operating_reestimate_fraction,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        n_test_frames=n_test_frames,
        coast_from_zero_mae=coast_from_zero_mae,
        c_train_flops=c_train_flops,
        c_reest_flops=c_reest_flops,
    )
    permutations = 2**n_seeds

    body: dict[str, Any] = {
        "schema": COUNT_REPRO_FE_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": COUNT_BED_ID,
        "reproduction_axis": REPRO_AXIS,
        "reproduces": "proof/STARSS23_COUNTING_BED.json",
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "question": (
            "does the counting bed's trained gate still reach lower coasted count-MAE than a rate-matched-"
            "random gate at a matched re-estimation budget when the frozen front-end AND the frozen count "
            "estimator are swapped for independently authored zero-parameter DSP rules"
        ),
        "varied_axes": {
            "featurizer": (
                "re-authored gammatone ERB filterbank plus causal relative spectral flux, replacing the "
                "sealed mel triangular filterbank plus half-wave log-mel difference"
            ),
            "featurizer_schema": COUNT_REPRO_FE_FEATURIZER_SCHEMA,
            "estimator": ESTIMATOR_RULE,
            "estimator_schema": COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
            "seed_family": list(REPRO_SEEDS),
        },
        "held_fixed": [
            "room-fold split (native fold-3 train, fold-4 test)",
            "count gate architecture and training objective",
            "coasted-count-MAE referee (pooled frame micro-average)",
            "controls (rate_matched_random, always_on, never_update, noisy_tv)",
            "exact sign-flip permutation and matched-budget harness",
        ],
        "metric": "coasted concurrent-source count MAE (lower is better), pooled frame micro-average",
        "metric_rule": METRIC_RULE,
        "cold_start": COLD_START,
        "direction": (
            "candidate < rate_matched_random (the trained gate places the same re-estimation budget better, "
            "reaching a lower count-MAE at matched re-estimation count)"
        ),
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_mae": round(sesoi.sesoi_mae, 12),
            "provisional": False,
            "selection_method": "cost-benefit one-test-clip change mass (reused rule), computed in code",
            "rationale": _sesoi_rationale(sesoi),
            "detail": sesoi.payload(),
            "cost_benefit": cb.payload(),
            "train_change_density": round(float(train_change_density), 12),
        },
        "operating_point_rule": (
            "the swept re-estimation budget whose fraction is closest to the train-set count-change density; "
            "a fixed rule set before scoring, using only train labels, never a val or test MAE argmax"
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
        "survival_criterion": (
            "SURVIVES only if, at the preregistered operating budget point, the candidate mean count-MAE is "
            "strictly below the rate-matched-random mean AND the mean paired delta (control minus candidate) "
            "is at least the registered SESOI AND the one-sided exact sign-flip p is at most 1/32; a tie is a "
            "null; a single reproduction can never set independent_scientific_confirmation"
        ),
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_frames": int(n_test_frames),
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
        },
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears alpha AND "
            "at least three bias-independent reproductions triangulate the same direction; a single run can "
            "never promote, and independent_scientific_confirmation is set only by the separate verifier"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body

DEFAULT_REPRO_PREREG_PATH = Path("proof/STARSS23_COUNTING_REPRO_featurizer_estimator.prereg.json")
