
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

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_superflux_spectral.prereg.json")

FEATURIZER_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "superflux_spectral",
        "hypothesis": (
            "A frozen SuperFlux front-end (mu-law log-mel with a frequency-max-filtered, vibrato-"
            "suppressed positive spectral flux) is a stronger hand-crafted onset detector than the base "
            "half-wave-rectified flux, so it exposes sharper onset structure the unchanged gate can "
            "localize, raising onset F1 over rate-matched-random at matched budget."
        ),
    },
    {
        "variant_id": "spatial_doa_intensity",
        "hypothesis": (
            "A frozen active-intensity direction-of-arrival front-end (the FOA acoustic intensity vector "
            "per band) gives the gate a spatial onset cue absent from the omnidirectional base flux, so a "
            "direction change at an onset separates events the base front-end blurs, beating random "
            "placement."
        ),
    },
    {
        "variant_id": "harmonic_comb_flux",
        "hypothesis": (
            "A frozen harmonic-comb-filtered flux front-end (spectral flux after a fixed comb emphasis of "
            "harmonic partials) suppresses broadband loudness swells and fires the gate on pitched-onset "
            "attacks instead of loud steady passages, beating uniform random placement at matched budget."
        ),
    },
)


class FeaturizersPreregRefusal(ValueError):
    pass


_FAMILY_PREREG = {
    "schema": FEATURIZERS_PREREG_SCHEMA,
    "wave": "F1 frozen-featurizer iteration",
    "members_field": "variants",
    "member_id_field": "variant_id",
    "member_label": "featurizer",
    "family_phrase": "three frozen featurizers",
    "n_field": "n_variants",
    "per_alpha_field": "per_variant_alpha",
    "alpha_digits": 6,
    "statistic": "mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
    "promotion_bar": (
        "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
        "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
        "same direction; a single run, and any run at n equals 5 across this three-featurizer family, "
        "can never promote"
    ),
    "refusal": FeaturizersPreregRefusal,
    "empty_message": "at least one frozen featurizer must be preregistered",
    "duplicate_message": "featurizer ids must be unique",
    "malformed_message": "each featurizer needs a variant_id and a one-line hypothesis",
    "extra": {
        "only_frozen_front_end_differs": True,
        "gate_unchanged": (
            "the same sealed 264-input candidate gate (3193 trained parameters, few-KB online state) reads "
            "every featurizer; no featurizer adds a trained parameter, and each emits exactly 256 features"
        ),
    },
}


def build_featurizers_prereg(
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
    variants: tuple[dict[str, str], ...] = FEATURIZER_VARIANTS,
    base_prereg_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    options = locals()
    options["members"] = options.pop("variants")
    return build_family_prereg(**options, **_FAMILY_PREREG)


def structural_facts_from_superflux_cache(
    *,
    cache_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> StructuralFacts:

    from .feature_cache import DEFAULT_CACHE_ROOT, load_or_build_cached_corpus

    corpus = load_or_build_cached_corpus(
        front_end="superflux", cache_root=cache_root or DEFAULT_CACHE_ROOT
    )
    rates = target_rates or RealBedConfig().target_rates
    return StructuralFacts.from_corpus(corpus, rates)
