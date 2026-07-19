
from __future__ import annotations

from pathlib import Path
from typing import Any

from .prereg import (
    DEFAULT_C_DOWN_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    PREREGISTERED_SESOI_F1,
    StructuralFacts,
    build_family_prereg,
)
from .real_artifact import RealBedConfig

VARIANTS_PREREG_SCHEMA = "mop-starss23-escs-bed-variants-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_VARIANTS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_VARIANTS.prereg.json")

GATE_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "refractory_nms",
        "hypothesis": (
            "A post-decision refractory window (collar-width non-maximum suppression on p_fire) forbids "
            "a second fire inside an already-covered collar, so the fixed budget spreads to distinct "
            "onsets instead of clustering adjacent fires on one high-energy region, raising distinct-"
            "onset recall over rate-matched-random at matched budget."
        ),
    },
    {
        "variant_id": "flux_novelty_target",
        "hypothesis": (
            "Retraining the value-of-computation target on local flux novelty (a frame's flux minus its "
            "causal neighborhood median) rather than absolute energy teaches the gate to fire on change "
            "points, not loud steady regions, so its placement beats free random placement."
        ),
    },
    {
        "variant_id": "energy_whitened_features",
        "hypothesis": (
            "Partialling out raw magnitude by dividing the features by a causal running-energy EMA before "
            "the gate removes the loud-region confound, so a sustained high-energy passage stops "
            "repeatedly attracting fires and the budget lands on separated onsets."
        ),
    },
    {
        "variant_id": "recency_gap_penalty",
        "hypothesis": (
            "A recurrence-aware online feature that penalizes firing shortly after a prior fire (the "
            "frames-since-last-fire gap in the online state) makes the gate withhold adjacent fires and "
            "distribute the budget across separated onsets, beating uniform random placement."
        ),
    },
)


class VariantsPreregRefusal(ValueError):
    pass


_FAMILY_PREREG = {
    "schema": VARIANTS_PREREG_SCHEMA,
    "wave": "E1 gate-variant iteration",
    "members_field": "variants",
    "member_id_field": "variant_id",
    "member_label": "variant",
    "family_phrase": "four variants",
    "n_field": "n_variants",
    "per_alpha_field": "per_variant_alpha",
    "alpha_digits": 4,
    "statistic": "mean of paired per-seed F1 deltas (variant minus rate_matched_random)",
    "promotion_bar": (
        "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
        "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
        "same direction; a single run, and any run at n equals 5 across this four-variant family, can "
        "never promote"
    ),
    "refusal": VariantsPreregRefusal,
    "empty_message": "at least one gate variant must be preregistered",
    "duplicate_message": "variant ids must be unique",
    "malformed_message": "each variant needs a variant_id and a one-line hypothesis",
}


def build_variants_prereg(
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
    variants: tuple[dict[str, str], ...] = GATE_VARIANTS,
    base_prereg_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    options = locals()
    options["members"] = options.pop("variants")
    return build_family_prereg(**options, **_FAMILY_PREREG)


def structural_facts_from_cache(
    *,
    cache_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> StructuralFacts:

    from .feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus

    corpus = load_cached_corpus(cache_root=cache_root or DEFAULT_CACHE_ROOT)
    rates = target_rates or RealBedConfig().target_rates
    return StructuralFacts.from_corpus(corpus, rates)
