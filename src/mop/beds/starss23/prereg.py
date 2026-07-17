"""Preregistration of the STARSS23 ESCS bed SESOI and analysis plan.

This is a net-new, additive component. It fixes the smallest-effect-size-of-interest (SESOI) on onset F1
by explicit cost-benefit reasoning (docs/ESCS_DEEP_RESEARCH.md Q5a option 3) and freezes the whole
analysis plan into a self-sealed artifact ``proof/STARSS23_ESCS_BED.prereg.json`` that MUST be written
before the run reads any test-split score. The committed provisional SESOI (0.05) was a placeholder; this
module replaces it with a preregistered value carrying a written, quantified rationale.

Nothing here reads a test score. The cost-benefit derivation uses only fixed compute anchors (the gate's
amortized C_train and the downstream cost per firing) and structural facts of the corpus that are known
from the labels before any arm is scored (the test-clip count, the pooled test onset count, and the
train-set onset density). The sealed body carries ``activation_allowed=false`` and
``scientific_promotion=false`` and a fixed timestamp that is passed in, never read from the wall clock
inside the sealed body.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID, CLAIM_SCOPE
from .gate import C_TRAIN_ANCHOR
from .schema import COLLAR_FRAMES, FRAME_MS
from .stats import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS

PREREG_SCHEMA = "mop-starss23-escs-bed-prereg/v1"
STAGE = 3

# The metric and direction are fixed by the bed and restated here so the prereg is self-contained.
PREREG_METRIC = "onset F1 at DCASE plus or minus 200 ms collar, greedy one-to-one, strict point-wise PR"
PREREG_DIRECTION = "candidate > rate_matched_random (candidate places the same firing budget better)"

# The amortized training cost anchor from the recipe: C_train = 8 * 54000 * 3 * 6385 = 8.27e9 FLOPs.
DEFAULT_C_TRAIN_FLOPS = C_TRAIN_ANCHOR
# The downstream cost of one fired frame (the expensive matched decision the gate decides to spend).
DEFAULT_C_DOWN_FLOPS = 40_000

# The preregistered SESOI on onset F1, fixed below by the cost-benefit derivation.
PREREGISTERED_SESOI_F1 = 0.05

N_PAIRED_SEEDS = 5


class PreregRefusal(ValueError):
    """Raised when a preregistration input is malformed."""


def _require_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PreregRefusal(f"{label} must be a finite number")
    if float(value) <= 0.0:
        raise PreregRefusal(f"{label} must be positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class CostBenefit:
    """The cost-benefit inputs and derived break-even that justify the SESOI on the F1 scale."""

    c_train_flops: int
    c_down_flops: int
    operating_firing_fraction: float
    per_frame_saving_flops: float
    break_even_frames: float
    break_even_hours_audio: float
    train_flops_in_firing_equivalents: float
    n_test_clips: int
    n_test_onsets: int
    per_onset_f1_granularity: float
    one_test_clip_f1_scale: float

    def payload(self) -> dict[str, Any]:
        return {
            "c_train_flops": self.c_train_flops,
            "c_down_flops": self.c_down_flops,
            "operating_firing_fraction": round(self.operating_firing_fraction, 12),
            "per_frame_saving_flops": round(self.per_frame_saving_flops, 6),
            "break_even_frames": round(self.break_even_frames, 3),
            "break_even_hours_audio": round(self.break_even_hours_audio, 4),
            "train_flops_in_firing_equivalents": round(self.train_flops_in_firing_equivalents, 3),
            "n_test_clips": self.n_test_clips,
            "n_test_onsets": self.n_test_onsets,
            "per_onset_f1_granularity": round(self.per_onset_f1_granularity, 9),
            "one_test_clip_f1_scale": round(self.one_test_clip_f1_scale, 9),
        }


def compute_cost_benefit(
    *,
    c_train_flops: int,
    c_down_flops: int,
    operating_firing_fraction: float,
    n_test_clips: int,
    n_test_onsets: int,
) -> CostBenefit:
    """Derive the break-even and F1 granularity that bound the SESOI. Reads no test score."""

    c_train_flops = int(_require_positive(c_train_flops, "c_train_flops"))
    c_down_flops = int(_require_positive(c_down_flops, "c_down_flops"))
    rho = _require_positive(operating_firing_fraction, "operating_firing_fraction")
    if not 0.0 < rho < 1.0:
        raise PreregRefusal("operating_firing_fraction must be strictly between 0 and 1")
    if n_test_clips <= 0 or n_test_onsets <= 0:
        raise PreregRefusal("n_test_clips and n_test_onsets must be positive")

    # Per-frame downstream FLOPs saved against the always-on reference at the operating firing fraction.
    per_frame_saving = (1.0 - rho) * c_down_flops
    # Break-even query count N* = C_train / per-query saving (docs Q2e, Q5a option 3).
    break_even = c_train_flops / per_frame_saving
    # 10 label frames per second, 3600 s per hour.
    break_even_hours = break_even / 10.0 / 3600.0
    firing_equivalents = c_train_flops / c_down_flops
    # One recovered onset moves pooled recall (and so F1) by about 1 / n_test_onsets: the measurement
    # granularity of the bed. One test clip is about n_test_onsets / n_test_clips onsets, the smallest
    # unit of the experimental design (the clip is the experimental unit).
    per_onset_granularity = 1.0 / n_test_onsets
    one_clip_scale = (n_test_onsets / n_test_clips) / n_test_onsets

    return CostBenefit(
        c_train_flops=c_train_flops,
        c_down_flops=c_down_flops,
        operating_firing_fraction=rho,
        per_frame_saving_flops=per_frame_saving,
        break_even_frames=break_even,
        break_even_hours_audio=break_even_hours,
        train_flops_in_firing_equivalents=firing_equivalents,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
        per_onset_f1_granularity=per_onset_granularity,
        one_test_clip_f1_scale=one_clip_scale,
    )


def _sesoi_rationale(cb: CostBenefit, sesoi: float) -> str:
    return (
        "Cost-benefit SESOI on onset F1. The candidate and the rate-matched-random control fire the same "
        "count K at equal FLOPs, so the amortized training cost C_train "
        f"({cb.c_train_flops} FLOPs, equal to {cb.train_flops_in_firing_equivalents:.0f} downstream-firing "
        "equivalents at C_down = "
        f"{cb.c_down_flops} FLOPs) buys nothing but the F1 advantage of learned placement over free random "
        "placement. At the operating firing fraction "
        f"{cb.operating_firing_fraction:.3f} the per-frame downstream saving against always-on is "
        f"{cb.per_frame_saving_flops:.0f} FLOPs, so the gate does not even repay C_train until "
        f"N* = {cb.break_even_frames:,.0f} frames (about {cb.break_even_hours_audio:.1f} hours of audio). "
        "Given that deployment scale, the smallest F1 win worth registering is one that is both "
        "measurable above the pseudoreplication floor and economically meaningful. One recovered onset "
        f"moves pooled F1 by about {cb.per_onset_f1_granularity:.4f} (1 / {cb.n_test_onsets} test onsets), "
        "and one test clip (the experimental unit) is about "
        f"{cb.n_test_onsets / cb.n_test_clips:.1f} onsets, i.e. an F1 scale of "
        f"{cb.one_test_clip_f1_scale:.4f}. The chosen SESOI of "
        f"{sesoi:.2f} sits an order of magnitude above the per-onset granularity (so it is not "
        "measurement noise) and at about one test clip's worth of correctly-placed onsets (so a win below "
        "it recovers less than one clip of onsets more than free random firing and does not justify "
        "carrying a trained module over the zero-training control). A win below the SESOI is not "
        "promotable even if the one-sided sign-flip p clears alpha."
    )


def build_prereg(
    *,
    timestamp: str,
    operating_firing_fraction: float,
    n_test_clips: int,
    n_test_onsets: int,
    train_onset_density: float,
    n_test_frames: int,
    sesoi_f1: float = PREREGISTERED_SESOI_F1,
    c_train_flops: int = DEFAULT_C_TRAIN_FLOPS,
    c_down_flops: int = DEFAULT_C_DOWN_FLOPS,
    n_seeds: int = N_PAIRED_SEEDS,
) -> dict[str, Any]:
    """Assemble the self-sealed preregistration body. The timestamp is passed, never read from a clock."""

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise PreregRefusal("timestamp must be a non-empty string passed by the caller")
    sesoi = _require_positive(sesoi_f1, "sesoi_f1")
    if n_seeds <= 0:
        raise PreregRefusal("n_seeds must be positive")

    cb = compute_cost_benefit(
        c_train_flops=c_train_flops,
        c_down_flops=c_down_flops,
        operating_firing_fraction=operating_firing_fraction,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
    )
    permutations = 2**n_seeds

    body: dict[str, Any] = {
        "schema": PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "metric": PREREG_METRIC,
        "collar_frames": COLLAR_FRAMES,
        "collar_ms": COLLAR_FRAMES * FRAME_MS,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_f1": round(sesoi, 12),
            "provisional": False,
            "selection_method": "cost-benefit (docs/ESCS_DEEP_RESEARCH.md Q5a option 3)",
            "rationale": _sesoi_rationale(cb, sesoi),
            "cost_benefit": cb.payload(),
            "train_onset_density": round(float(train_onset_density), 12),
        },
        "operating_point_rule": (
            "the swept firing budget whose firing fraction is closest to the train-set onset density; "
            "a fixed rule set before scoring, using only train labels, never a val or test F1 argmax"
        ),
        "sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "statistic": "mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
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
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears "
            "alpha AND at least three bias-independent reproductions triangulate the same direction; a "
            "single run can never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


DEFAULT_PREREG_PATH = Path("proof/STARSS23_ESCS_BED.prereg.json")


def write_prereg(body: dict[str, Any], out_path: str | Path = DEFAULT_PREREG_PATH) -> Path:
    """Write the self-sealed preregistration as canonical JSON bytes so its on-disk digest is stable."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path
