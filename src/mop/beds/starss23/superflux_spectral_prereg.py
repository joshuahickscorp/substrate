"""Sealed preregistration for the F1 frozen-featurizer iteration wave (STARSS23 ESCS bed).

This is a net-new, additive component. It changes no sealed scoring logic. The first real Stage-3 run
nulled because the trained value-of-computation gate, reading the base half-wave-rectified log-mel flux,
clusters its fires on high-energy regions and recovers fewer distinct onsets than uniform random
placement at matched budget. The F1 wave keeps the SAME sealed gate, referee, and controls and swaps only
the FROZEN zero-trained-parameter FEATURIZER, testing whether a stronger hand-crafted onset front-end
exposes onset structure the unchanged gate can localize.

Testing a family of frozen featurizers against one fixed test split inflates the family-wise error, so
this module preregisters, IN CODE and sealed to ``proof/STARSS23_ESCS_BED_superflux_spectral.prereg.json``
BEFORE any featurizer reads a test score:

- the metric and direction (onset F1 at the DCASE plus-or-minus 200 ms collar, greedy one-to-one, strict
  point-wise PR; direction candidate greater than rate-matched-random), restated from the committed prereg;
- the SESOI (0.05), on the identical cost-benefit basis as the committed prereg, recomputed from the
  label structure of the corpus (never from a test score);
- the exact one-sided sign-flip plan (five paired seeds, min one-sided p 1/32, two-sided 0.05 unreachable
  at n equals 5);
- a multiplicity control across the three featurizer families (Bonferroni). With three featurizers the
  per-family adjusted alpha is 0.05/3 = 0.016667, and the smallest achievable one-sided sign-flip p at
  five paired seeds is 1/32 = 0.03125, which exceeds it, so NO single featurizer can clear family-wise
  significance from this family alone at n equals 5: a preregistered statistical wall, not a moved
  goalpost;
- the claim ceiling (the clip is the experimental unit, the verb is bounded to "consistent with");
- the three featurizer ids, each with a one-line falsifiable hypothesis.

Nothing here reads a test-split F1. The cost-benefit derivation uses only fixed compute anchors and label
structure known before scoring (test-clip count, pooled test onset count, train-set onset density). The
timestamp is passed by the caller, never read from the wall clock inside the sealed body, and the sealed
body carries ``activation_allowed=false`` and ``scientific_promotion=false``.

House style: no em dashes and no en dashes.
"""

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
    family_analysis_plan,
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
    """Raised when the featurizer preregistration inputs are malformed."""


def _multiplicity_block(n_variants: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:
    """Bonferroni family-wise control across the featurizer family, honest about the n equals 5 floor."""

    per_variant_alpha = alpha / n_variants
    return {
        "n_variants": n_variants,
        "correction": "Bonferroni",
        "family_alpha": alpha,
        "per_variant_alpha": round(per_variant_alpha, 12),
        "min_achievable_one_sided_p": round(min_one_sided_p, 12),
        "family_significance_reachable_at_n5": bool(min_one_sided_p <= per_variant_alpha),
        "rationale": (
            "three frozen featurizers scored against one fixed test split inflate the family-wise error, "
            f"so each featurizer is held to the Bonferroni-adjusted alpha {per_variant_alpha:.6f}. With "
            f"five paired seeds the smallest achievable one-sided sign-flip p is {min_one_sided_p:.5f}, "
            "which exceeds that adjusted alpha, so no single featurizer can clear family-wise significance "
            "from this family alone. This is a preregistered statistical wall: a featurizer may still "
            "register a SESOI-exceeding effect for triangulation, but family-wise promotion needs more "
            "seeds and bias-independent reproductions, never a larger claim squeezed from n equals 5"
        ),
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
    """Assemble the self-sealed featurizer preregistration body.

    The timestamp is passed by the caller and never read from a clock.
    """

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
    """Derive the label-only structural facts from the built SuperFlux cache. Reads no test-split F1.

    Onset counts, frame counts, and the train onset density are label-only and identical across
    featurizers (only the feature bytes differ), so reading them from the SuperFlux cache is a pure
    label-structure read. The operating firing fraction is the swept target rate closest to the train
    onset density, the exact rule the producer preregisters, using train labels only.
    """

    from .feature_cache import DEFAULT_CACHE_ROOT, load_or_build_cached_corpus

    corpus = load_or_build_cached_corpus(
        front_end="superflux", cache_root=cache_root or DEFAULT_CACHE_ROOT
    )
    rates = target_rates or RealBedConfig().target_rates
    train_density = corpus.train_onset_density()
    operating_rate = min(rates, key=lambda r: abs(r - train_density))
    return StructuralFacts(
        operating_firing_fraction=float(operating_rate),
        n_test_clips=corpus.n_test_clips(),
        n_test_onsets=corpus.n_test_onsets(),
        train_onset_density=float(train_density),
        n_test_frames=corpus.n_test_frames(),
    )


def _main(argv: list[str] | None = None) -> int:
    """Seal the featurizer preregistration from the cached corpus structural facts and print its digest."""

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
        operating_firing_fraction=facts.operating_firing_fraction,
        n_test_clips=facts.n_test_clips,
        n_test_onsets=facts.n_test_onsets,
        train_onset_density=facts.train_onset_density,
        n_test_frames=facts.n_test_frames,
        base_prereg_canonical_sha256=base_prereg_digest(),
    )
    path = write_canonical_json(body, args.out)
    print(f"wrote {path}")
    print(
        json.dumps(
            {
                "canonical_sha256": body["canonical_sha256"],
                "sesoi_f1": body["sesoi"]["sesoi_f1"],
                "n_variants": body["multiplicity"]["n_variants"],
                "family_significance_reachable_at_n5": body["multiplicity"][
                    "family_significance_reachable_at_n5"
                ],
                "variant_ids": [v["variant_id"] for v in body["variants"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
