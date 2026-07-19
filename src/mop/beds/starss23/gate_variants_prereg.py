
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

VARIANTS_PREREG_SCHEMA = "mop-starss23-escs-bed-variants-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_VARIANTS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_VARIANTS.prereg.json")

# The four E1 gate variants. Each is a distinct, falsifiable answer to the diagnosed failure: the base
# gate clusters its fires on high-energy regions instead of spreading them across distinct onsets. Each
# hypothesis is one line, scored against the same sealed referee and rate-matched-random control.
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


def _multiplicity_block(n_variants: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:

    return bonferroni_family(
        n_variants, min_one_sided_p, alpha,
        family_phrase="four variants", member_label="variant",
        n_field="n_variants", per_alpha_field="per_variant_alpha", alpha_digits=4,
    )


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

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise VariantsPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise VariantsPreregRefusal("n_seeds must be positive")
    if not variants:
        raise VariantsPreregRefusal("at least one gate variant must be preregistered")
    ids = [entry["variant_id"] for entry in variants]
    if len(set(ids)) != len(ids):
        raise VariantsPreregRefusal("variant ids must be unique")
    for entry in variants:
        if not entry.get("variant_id") or not entry.get("hypothesis"):
            raise VariantsPreregRefusal("each variant needs a variant_id and a one-line hypothesis")

    body: dict[str, Any] = {
        "schema": VARIANTS_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "E1 gate-variant iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
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
            statistic="mean of paired per-seed F1 deltas (variant minus rate_matched_random)",
            multiplicity=_multiplicity_block,
        ),
        "variants": [
            {"variant_id": entry["variant_id"], "hypothesis": entry["hypothesis"]}
            for entry in variants
        ],
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
            "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
            "same direction; a single run, and any run at n equals 5 across this four-variant family, can "
            "never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


def structural_facts_from_cache(
    *,
    cache_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> StructuralFacts:

    from .feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus

    corpus = load_cached_corpus(cache_root=cache_root or DEFAULT_CACHE_ROOT)
    rates = target_rates or RealBedConfig().target_rates
    return StructuralFacts.from_corpus(corpus, rates)


def _main(argv: list[str] | None = None) -> int:

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Seal the STARSS23 ESCS gate-variant preregistration.")
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--out", default=str(DEFAULT_VARIANTS_PREREG_PATH))
    args = parser.parse_args(argv)

    facts = structural_facts_from_cache(cache_root=args.cache_root)
    body = build_variants_prereg(
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
