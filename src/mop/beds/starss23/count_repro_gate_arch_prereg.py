
from __future__ import annotations

from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE
from .count_estimator import FLOPS_PER_REESTIMATE
from .count_prereg import (
    DEFAULT_C_TRAIN_FLOPS,
    compute_count_cost_benefit,
)
from .count_referee import COLD_START, METRIC_RULE
from .count_repro_gate_arch_gate import (
    D_IN_GATE_ARCH,
    HIDDEN1,
    HIDDEN2,
    N_OUT,
    PARAM_CEILING,
    REPRO_AXIS,
    param_count_two_layer,
)
from .experiments import COUNT_BED_ID

COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA = "mop-starss23-count-repro-gate-arch-prereg/v1"
STAGE = 3

DEFAULT_C_REEST_FLOPS = FLOPS_PER_REESTIMATE

MIN_GRANULARITY_MULTIPLE = 100.0

N_PAIRED_SEEDS = 5

PREREG_METRIC = "coasted concurrent-source count MAE (lower is better), pooled frame micro-average"
PREREG_DIRECTION = (
    "candidate < rate_matched_random (the re-authored two-layer gate places the same re-estimation budget "
    "better, reaching a lower count-MAE at matched re-estimation count)"
)


class CountReproGateArchPreregRefusal(ValueError):
    pass


def _sesoi_rationale(cb: Any, sesoi: float) -> str:
    granularity_multiple = sesoi / cb.per_frame_granularity
    sesoi_in_frame_errors = sesoi * cb.n_test_frames
    return (
        "Cost-benefit SESOI on coasted count-MAE, reused unchanged from the sealed counting bed and "
        "recomputed here from this reproduction's own label-only structural facts. The registered number "
        f"is one test clip's catchable change mass: the fold carries {cb.n_test_changes} count changes over "
        f"{cb.n_test_clips} test clips ({cb.changes_per_clip:.1f} per clip) with a mean run of "
        f"{cb.mean_run_frames:.1f} frames, so one clip's catchable change mass is about "
        f"{cb.one_clip_change_mass_frames:.0f} frame-errors, which on the pooled {cb.n_test_frames}-frame "
        f"count-MAE scale is {sesoi:.6f} (exactly 0.5 / n_test_clips). This is a property of the ground "
        "truth labels alone and does not depend on the gate architecture, so varying the gate shape leaves "
        f"it unchanged. It sits at {granularity_multiple:.0f} times the pooled per-frame granularity "
        f"({cb.per_frame_granularity:.2e}, one frame-error over {cb.n_test_frames} frames), about "
        f"{sesoi_in_frame_errors:.0f} frame-errors, far above measurement noise. A count-MAE win below this "
        "SESOI recovers less than about one clip of change tracking over free rate-matched-random placement "
        "and is not promotable even if the one-sided sign-flip p clears alpha."
    )


def build_count_repro_gate_arch_prereg(
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
        raise CountReproGateArchPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise CountReproGateArchPreregRefusal("n_seeds must be positive")

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
    floor = MIN_GRANULARITY_MULTIPLE * cb.per_frame_granularity
    if not sesoi > 0.0:
        raise CountReproGateArchPreregRefusal("the derived SESOI must be positive")
    if sesoi < floor:
        raise CountReproGateArchPreregRefusal(
            f"derived SESOI {sesoi:.3e} is below {MIN_GRANULARITY_MULTIPLE:.0f}x the per-frame granularity "
            f"floor {floor:.3e}; refusing a sub-floor effect size"
        )

    gate_arch_params = param_count_two_layer(D_IN_GATE_ARCH, HIDDEN1, HIDDEN2, N_OUT)
    if gate_arch_params > PARAM_CEILING:
        raise CountReproGateArchPreregRefusal(
            f"gate-architecture params {gate_arch_params} exceed the {PARAM_CEILING} ceiling"
        )

    permutations = 2**n_seeds
    body: dict[str, Any] = {
        "schema": COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": COUNT_BED_ID,
        "reproduction_axis": REPRO_AXIS,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "question": (
            "gate-architecture reproduction: does a differently-shaped trained gate (two hidden layers, "
            "264 -> 8 -> 4 -> 1) reach lower coasted count-MAE at the same re-estimation budget as a "
            "rate-matched-random gate, holding every other axis of the sealed counting bed fixed"
        ),
        "varied_axis": {
            "axis": REPRO_AXIS,
            "sealed_gate": "264 -> 12 -> 1 single hidden layer (3193 params)",
            "reproduction_gate": "264 -> 8 -> 4 -> 1 two hidden layers",
            "reproduction_gate_params": gate_arch_params,
            "param_ceiling": PARAM_CEILING,
            "seed_family": [40, 41, 42, 43, 44],
        },
        "held_fixed": [
            "room-disjoint split (native fold-3 train, fold-4 test)",
            "frozen count featurizer",
            "frozen count estimator",
            "coasted-count-MAE referee (pooled frame micro-average)",
            "rate_matched_random, always_on, never_update, noisy_tv controls",
            "8 self-derived online-state scalars and the value-of-computation targets",
            "exact sign-flip permutation over five paired seeds",
        ],
        "metric": PREREG_METRIC,
        "metric_rule": METRIC_RULE,
        "cold_start": COLD_START,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_mae": round(sesoi, 12),
            "provisional": False,
            "selection_method": (
                "cost-benefit one-test-clip change mass (0.5 / n_test_clips), reused unchanged from the "
                "sealed bed and recomputed from this reproduction's own labels; gate-architecture independent"
            ),
            "min_granularity_multiple": MIN_GRANULARITY_MULTIPLE,
            "granularity_multiple": round(sesoi / cb.per_frame_granularity, 6),
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
        "survive_criterion": (
            "at the preregistered operating budget point, the candidate strictly beats rate_matched_random "
            "in the same direction (lower count-MAE) AND the mean paired delta clears the registered SESOI "
            "AND the one-sided exact sign-flip p is at or below 1/32; a tie is a null"
        ),
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_frames": int(n_test_frames),
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
            "rationale": (
                f"with about {n_test_clips} test clips the clip is the experimental unit and frames are "
                "correlated sub-samples; a frame or clip bootstrap is refused and the claim verb is bounded "
                "to 'consistent with', never 'demonstrates' or 'significant'"
            ),
        },
        "promotion_bar": (
            "a single reproduction never promotes and never self-certifies; independent scientific "
            "confirmation is set only by the separately-authored verifier and requires at least three "
            "bias-independent reproductions plus human adjudication"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body

DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH = Path(
    "proof/STARSS23_COUNTING_REPRO_gate_arch.prereg.json"
)
