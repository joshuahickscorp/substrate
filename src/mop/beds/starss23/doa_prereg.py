
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.science.statistics import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS
from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE
from .doa_estimator import FLOPS_PER_REESTIMATE
from .doa_gate import ARCH_A_ID, ARCH_B_ID, C_TRAIN_ANCHOR_ARCH_A, C_TRAIN_ANCHOR_ARCH_B
from .doa_referee import DOA_COLD_START, METRIC_RULE
from .experiments import DOA_BED_ID
from .schema import FRAME_MS

DOA_PREREG_SCHEMA = "mop-starss23-doa-bed-prereg/v1"
STAGE = 3

PREREG_METRIC = "great-circle direction-of-arrival error in degrees (lower is better), clip-macro primary"
PREREG_DIRECTION = (
    "candidate < rate_matched_random on clip-macro great-circle MAE (the trained gate places the same "
    "re-estimation budget better than free random placement), independently under BOTH gate architectures"
)

DEFAULT_C_REEST_FLOPS = FLOPS_PER_REESTIMATE
N_PAIRED_SEEDS = 5

GRANULARITY_FLOOR_MULTIPLE_DOA = 10.0

PRIMARY_ALPHA = 0.05
ROOM_MAJORITY_ALPHA = 0.10

_FRAMES_PER_SECOND = 1000.0 / FRAME_MS  # 10 frames per second on the 100 ms label grid


class DoaPreregRefusal(ValueError):
    pass


def _require_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DoaPreregRefusal(f"{label} must be a finite number")
    if float(value) <= 0.0:
        raise DoaPreregRefusal(f"{label} must be positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class DoaCostBenefit:

    architecture: str
    c_train_flops: int
    c_reest_flops: int
    operating_reestimate_fraction: float
    per_reestimate_saving: float
    break_even_frames: float
    break_even_hours: float
    train_flops_in_reestimate_equivalents: float

    def payload(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "c_train_flops": self.c_train_flops,
            "c_reest_flops": self.c_reest_flops,
            "operating_reestimate_fraction": round(self.operating_reestimate_fraction, 12),
            "per_reestimate_saving": round(self.per_reestimate_saving, 6),
            "break_even_frames": round(self.break_even_frames, 3),
            "break_even_hours": round(self.break_even_hours, 4),
            "train_flops_in_reestimate_equivalents": round(self.train_flops_in_reestimate_equivalents, 3),
        }


def compute_doa_cost_benefit(
    *,
    architecture: str,
    c_train_flops_arch: int,
    c_reest_flops: int,
    operating_reestimate_fraction: float,
) -> DoaCostBenefit:

    c_train_flops_arch = int(_require_positive(c_train_flops_arch, "c_train_flops_arch"))
    c_reest_flops = int(_require_positive(c_reest_flops, "c_reest_flops"))
    rho = _require_positive(operating_reestimate_fraction, "operating_reestimate_fraction")
    if not 0.0 < rho < 1.0:
        raise DoaPreregRefusal("operating_reestimate_fraction must be strictly between 0 and 1")

    per_reestimate_saving = (1.0 - rho) * c_reest_flops
    break_even = c_train_flops_arch / per_reestimate_saving
    break_even_hours = break_even / _FRAMES_PER_SECOND / 3600.0
    reestimate_equivalents = c_train_flops_arch / c_reest_flops

    return DoaCostBenefit(
        architecture=architecture,
        c_train_flops=c_train_flops_arch,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=rho,
        per_reestimate_saving=per_reestimate_saving,
        break_even_frames=break_even,
        break_even_hours=break_even_hours,
        train_flops_in_reestimate_equivalents=reestimate_equivalents,
    )


@dataclass(frozen=True, slots=True)
class DoaClipLabelFact:

    clip_id: str
    n_active_frames: int
    n_changes: int


def compute_doa_sesoi_deg(
    test_clip_facts: Sequence[DoaClipLabelFact], mean_change_jump_deg: float
) -> tuple[float, list[dict[str, Any]]]:

    facts = tuple(test_clip_facts)
    if not facts:
        raise DoaPreregRefusal("at least one test clip fact is required")
    if isinstance(mean_change_jump_deg, bool) or not isinstance(mean_change_jump_deg, (int, float)):
        raise DoaPreregRefusal("mean_change_jump_deg must be a real number")
    jump = float(mean_change_jump_deg)
    if not math.isfinite(jump) or jump < 0.0:
        raise DoaPreregRefusal("mean_change_jump_deg must be a finite nonnegative number")

    n_clips = len(facts)
    values: list[float] = []
    brackets: list[dict[str, Any]] = []
    for fact in facts:
        if fact.n_active_frames <= 0:
            raise DoaPreregRefusal(f"clip {fact.clip_id!r} must have a positive active-frame count")
        if fact.n_changes <= 0:
            per_clip_mae_mass = 0.0
            mean_run = 0.0
        else:
            mean_run = fact.n_active_frames / fact.n_changes
            per_clip_mass_deg = jump * (mean_run / 2.0)
            per_clip_mae_mass = per_clip_mass_deg / fact.n_active_frames
        values.append(per_clip_mae_mass)
        brackets.append(
            {
                "clip_id": fact.clip_id,
                "n_active_frames": fact.n_active_frames,
                "n_changes": fact.n_changes,
                "mean_run_frames": round(mean_run, 6),
                "per_clip_mae_mass_deg": round(per_clip_mae_mass, 12),
            }
        )
    macro_restatement_deg = (math.fsum(values) / n_clips) / n_clips
    return macro_restatement_deg, brackets


def _sesoi_rationale(
    sesoi_deg: float,
    cb_a: DoaCostBenefit,
    cb_b: DoaCostBenefit,
    macro_granularity_deg: float,
    granularity_floor: float,
    n_test_clips: int,
    min_clip_active_frames: int,
) -> str:
    granularity_multiple = sesoi_deg / macro_granularity_deg if macro_granularity_deg > 0 else float("inf")
    return (
        "Cost-benefit and structural SESOI on clip-macro great-circle DoA MAE (degrees). The candidate and "
        "the rate-matched-random control spend the same re-estimation count K at equal FLOPs per "
        f"architecture, so the amortized training cost buys only the DoA-tracking advantage of learned "
        f"re-estimation placement over free random placement: Architecture A's C_train "
        f"({cb_a.c_train_flops} FLOPs, {cb_a.train_flops_in_reestimate_equivalents:.0f} re-estimation "
        f"equivalents) breaks even at about {cb_a.break_even_frames:,.0f} frames "
        f"({cb_a.break_even_hours:.1f} hours); Architecture B's C_train ({cb_b.c_train_flops} FLOPs, "
        f"{cb_b.train_flops_in_reestimate_equivalents:.0f} equivalents) breaks even at about "
        f"{cb_b.break_even_frames:,.0f} frames ({cb_b.break_even_hours:.1f} hours). The registered SESOI "
        f"is the architecture-independent structural number, used directly with no separate hardcoded "
        f"default it merely approximates: {sesoi_deg:.6f} degrees, one test clip's worth of a single "
        "missed direction change coasted for about half its typical inter-change dwell time, folded into "
        f"the equal-weight {n_test_clips}-clip macro mean by a double 1/n_clips normalization. The "
        f"clip-macro measurement granularity (one label-quantization degree of perturbation in the "
        f"shortest test clip, {min_clip_active_frames} active frames, carried into the macro mean) is "
        f"{macro_granularity_deg:.3e} degrees, so the SESOI sits at {granularity_multiple:.1f}x that floor, "
        f"a deliberately smaller multiple than the counting bed's 100x: angular error is bounded to "
        "[0, 180] degrees and the 1-degree label floor is already coarse relative to plausible "
        f"single-to-double-digit-degree SESOI magnitudes, so {GRANULARITY_FLOOR_MULTIPLE_DOA:.0f}x "
        "preserves the same qualitative discipline while staying achievable within the metric's real "
        "dynamic range. A win below the SESOI recovers less than about one clip of change-tracking over "
        "free random placement and is not promotable even if the clip-level sign-flip p clears alpha."
    )


def build_doa_prereg(
    *,
    timestamp: str,
    operating_reestimate_fraction: float,
    test_clip_facts: Sequence[DoaClipLabelFact],
    test_rooms: Sequence[str],
    train_change_density: float,
    mean_change_jump_deg: float,
    c_train_flops_arch_a: int = C_TRAIN_ANCHOR_ARCH_A,
    c_train_flops_arch_b: int = C_TRAIN_ANCHOR_ARCH_B,
    c_reest_flops: int = DEFAULT_C_REEST_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
) -> dict[str, Any]:

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise DoaPreregRefusal("timestamp must be a non-empty string passed by the caller")
    facts = tuple(test_clip_facts)
    if not facts:
        raise DoaPreregRefusal("at least one test clip fact is required")
    if n_seeds <= 0:
        raise DoaPreregRefusal("n_seeds must be positive")
    rooms = tuple(test_rooms)
    if not rooms:
        raise DoaPreregRefusal("at least one test room is required")

    cb_a = compute_doa_cost_benefit(
        architecture=ARCH_A_ID,
        c_train_flops_arch=c_train_flops_arch_a,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
    )
    cb_b = compute_doa_cost_benefit(
        architecture=ARCH_B_ID,
        c_train_flops_arch=c_train_flops_arch_b,
        c_reest_flops=c_reest_flops,
        operating_reestimate_fraction=operating_reestimate_fraction,
    )

    sesoi_deg, brackets = compute_doa_sesoi_deg(facts, mean_change_jump_deg)
    n_test_clips = len(facts)
    min_clip_active_frames = min(fact.n_active_frames for fact in facts)
    macro_granularity_deg = 1.0 / (n_test_clips * min_clip_active_frames)
    granularity_floor = GRANULARITY_FLOOR_MULTIPLE_DOA * macro_granularity_deg
    if not math.isfinite(sesoi_deg) or sesoi_deg <= 0.0:
        raise DoaPreregRefusal("the registered SESOI_DEG must be positive and finite")
    if sesoi_deg < granularity_floor:
        raise DoaPreregRefusal(
            f"registered SESOI_DEG {sesoi_deg} is below {GRANULARITY_FLOOR_MULTIPLE_DOA:.0f}x the "
            f"clip-macro granularity {macro_granularity_deg}; it would sit at the pseudoreplication floor"
        )

    n_clip_permutations = 2**n_test_clips
    n_seed_permutations = 2**n_seeds
    n_room_permutations = 2 ** len(rooms)

    body: dict[str, Any] = {
        "schema": DOA_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": DOA_BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "question": (
            "direction-of-arrival re-estimation value-of-computation: does a trained gate reach lower "
            "clip-macro great-circle DoA MAE at the same re-estimation budget as a rate-matched-random "
            "gate, independently under BOTH a single-hidden-layer and a two-hidden-layer gate architecture"
        ),
        "metric": PREREG_METRIC,
        "metric_rule": METRIC_RULE,
        "cold_start": list(DOA_COLD_START),
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "architectures": {"arch_a_id": ARCH_A_ID, "arch_b_id": ARCH_B_ID, "both_required": True},
        "sesoi": {
            "sesoi_deg": round(sesoi_deg, 12),
            "provisional": False,
            "selection_method": (
                "structural, architecture-independent: one test clip's worth of a single missed direction "
                "change, coasted half its typical inter-change dwell, folded into the clip-macro mean by a "
                "double 1/n_clips normalization; used directly, no separate hardcoded default parameter"
            ),
            "rationale": _sesoi_rationale(
                sesoi_deg, cb_a, cb_b, macro_granularity_deg, granularity_floor, n_test_clips,
                min_clip_active_frames,
            ),
            "cost_benefit_arch_a": cb_a.payload(),
            "cost_benefit_arch_b": cb_b.payload(),
            "per_clip_change_mass": brackets,
            "macro_granularity_deg": round(macro_granularity_deg, 12),
            "granularity_floor_multiple_required": GRANULARITY_FLOOR_MULTIPLE_DOA,
            "granularity_floor_deg": round(granularity_floor, 12),
            "granularity_multiple": round(sesoi_deg / macro_granularity_deg, 6),
            "clears_granularity_floor": sesoi_deg >= granularity_floor,
            "train_change_density": round(float(train_change_density), 12),
            "mean_change_jump_deg": round(float(mean_change_jump_deg), 12),
        },
        "operating_point_rule": (
            "the swept re-estimation budget whose fraction is closest to the train-set DoA change "
            "density; a fixed rule set before scoring, using only train labels, never a val or test MAE "
            "argmax; chosen independently per architecture from that architecture's own val probabilities"
        ),
        "primary_sign_flip_test_plan": {
            "test": "exact sign-flip permutation over clip-level paired deltas, meet in the middle, "
            "one-sided upper tail, run independently under BOTH architectures",
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_permutations": n_clip_permutations,
            "statistic": (
                "sum of per-clip paired deltas (rate_matched_random minus candidate, each clip's own "
                "value averaged over the paired seeds first)"
            ),
            "min_one_sided_p": round(1.0 / n_clip_permutations, 15),
            "alpha": PRIMARY_ALPHA,
        },
        "secondary_sign_flip_test_plan": {
            "test": "exact sign-flip permutation over the 5 paired-seed clip-macro means, one-sided upper "
            "tail, REQUIRED TO AGREE IN DIRECTION under both architectures (my own added discipline, not "
            "silently dropped)",
            "experimental_unit": "paired seed",
            "n_paired_seeds": n_seeds,
            "n_permutations": n_seed_permutations,
            "min_one_sided_p": round(1.0 / n_seed_permutations, 12),
            "two_sided_floor": round(2.0 / n_seed_permutations, 12),
            "two_sided_alpha_reachable": (2.0 / n_seed_permutations) <= 0.05,
            "phipson_smyth_applied": False,
            "required_to_agree_in_direction": True,
            "direction_agreement_rule": "mean of the 5 paired-seed clip-macro deltas strictly positive",
        },
        "room_majority_plan": {
            "test": (
                "calibration-note discipline (additive, not in the literal design spec): collapse each "
                "room's within-room clip deltas to a majority vote, exact binomial-style sign-flip over "
                "rooms; any clip-level p below about the room-level floor is treated with default "
                "suspicion of within-room pseudoreplication and cross-checked against this readout"
            ),
            "experimental_unit": "room (majority vote of within-room clips)",
            "n_test_rooms": len(rooms),
            "test_rooms": list(rooms),
            "n_permutations": n_room_permutations,
            "min_one_sided_p": round(1.0 / n_room_permutations, 12),
            "does_not_contradict_alpha": ROOM_MAJORITY_ALPHA,
            "required": True,
        },
        "promotion_rule": (
            "for architecture X, SURVIVES(X) iff: (1) candidate's clip-macro macro_mae_deg is strictly "
            "lower than rate_matched_random's (mean over 5 seeds); AND (2) the mean clip-level paired "
            "delta (each clip's own value averaged over the 5 seeds first, then meaned over clips) meets "
            "or exceeds the registered SESOI_DEG; AND (3) the primary clip-level exact sign-flip one-sided "
            "p clears alpha=0.05; AND (4, added discipline) the 5-seed secondary sign-flip agrees in "
            "direction; AND (5, calibration-note discipline) the room-majority one-sided p does not exceed "
            f"{ROOM_MAJORITY_ALPHA}. Any tie in clause 1 is a null for that architecture. Bed-level: "
            "SURVIVES(arch_a) AND SURVIVES(arch_b) is reportable as 'consistent with', flagged for human "
            "review only; exactly one SURVIVES is 'architecture-fragile', not a survive; neither is null. "
            "independent_scientific_confirmation is never set true by producer or verifier on one run."
        ),
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
            "rationale": (
                f"with {n_test_clips} test clips across {len(rooms)} rooms the clip is the experimental "
                "unit and frames are correlated sub-samples within a clip, and clips within a room may "
                "themselves be correlated (the room-majority readout above); the claim verb is bounded to "
                "'consistent with', never 'demonstrates' or 'significant'"
            ),
        },
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body

DEFAULT_DOA_PREREG_PATH = Path("proof/STARSS23_DOA_BED.prereg.json")
