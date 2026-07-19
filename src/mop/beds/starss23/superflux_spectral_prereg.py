
from __future__ import annotations

from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_sha256, write_canonical_json

from . import BED_ID, CLAIM_SCOPE
from .prereg import (
    DEFAULT_C_DOWN_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    PREREGISTERED_SESOI_F1,
    StructuralFacts,
    base_prereg_digest,
    bonferroni_family,
    family_analysis_plan,
    family_cli_summary,
)
from .real_artifact import RealBedConfig

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_superflux_spectral.prereg.json")

# The three F1 frozen-featurizer families. Each is a distinct, falsifiable answer to the diagnosed
# failure: the base half-wave-rectified log-mel flux is a weak onset front-end whose novelty the gate
# clusters on loud regions. Each keeps the SAME sealed gate and controls; only the frozen front-end
# differs. superflux_spectral is the featurizer run in this wave; the other two are its preregistered
# family siblings, so the multiplicity wall is honest.
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


def _multiplicity_block(n_variants: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:

    return bonferroni_family(
        n_variants, min_one_sided_p, alpha,
        family_phrase="three frozen featurizers", member_label="featurizer",
        n_field="n_variants", per_alpha_field="per_variant_alpha", alpha_digits=6,
    )


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

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise FeaturizersPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise FeaturizersPreregRefusal("n_seeds must be positive")
    if not variants:
        raise FeaturizersPreregRefusal("at least one frozen featurizer must be preregistered")
    ids = [entry["variant_id"] for entry in variants]
    if len(set(ids)) != len(ids):
        raise FeaturizersPreregRefusal("featurizer ids must be unique")
    for entry in variants:
        if not entry.get("variant_id") or not entry.get("hypothesis"):
            raise FeaturizersPreregRefusal("each featurizer needs a variant_id and a one-line hypothesis")

    body: dict[str, Any] = {
        "schema": FEATURIZERS_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "F1 frozen-featurizer iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
        "only_frozen_front_end_differs": True,
        "gate_unchanged": (
            "the same sealed 264-input candidate gate (3193 trained parameters, few-KB online state) reads "
            "every featurizer; no featurizer adds a trained parameter, and each emits exactly 256 features"
        ),
        **family_analysis_plan(
            operating_firing_fraction=operating_firing_fraction,
            n_test_clips=n_test_clips,
            n_test_onsets=n_test_onsets,
            train_onset_density=train_onset_density,
            n_test_frames=n_test_frames,
            sesoi_f1=sesoi_f1,
            c_train_flops=c_train_flops,
            c_down_flops=c_down_flops,
            n_seeds=n_seeds,
            n_members=len(variants),
            statistic="mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
            multiplicity=_multiplicity_block,
        ),
        "variants": [
            {"variant_id": entry["variant_id"], "hypothesis": entry["hypothesis"]}
            for entry in variants
        ],
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
            "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
            "same direction; a single run, and any run at n equals 5 across this three-featurizer family, "
            "can never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


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


def _main(argv: list[str] | None = None) -> int:

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Seal the STARSS23 ESCS frozen-featurizer preregistration.")
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--out", default=str(DEFAULT_FEATURIZERS_PREREG_PATH))
    args = parser.parse_args(argv)

    facts = structural_facts_from_superflux_cache(cache_root=args.cache_root)
    body = build_featurizers_prereg(
        timestamp=args.timestamp,
        **facts.payload(),
        base_prereg_canonical_sha256=base_prereg_digest(),
    )
    path = write_canonical_json(body, args.out)
    print(f"wrote {path}")
    print(
        json.dumps(
            family_cli_summary(body, "n_variants", "variants", "variant_id", "variant_ids"),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
