"""Component 9b: the sealed preregistration for the E1 gate-variant iteration wave.

This is a net-new, additive component. It changes no sealed scoring logic. The first real Stage-3 run
nulled because the trained value-of-computation gate clusters roughly 42 percent of its fires adjacently
on high-energy regions and so recovers fewer distinct onsets than uniform random placement at matched
budget (docs/mixture_of_perspectives/26_escs_starss23_bed.md). The E1 wave tests four new gate designs
against the SAME sealed referee and controls, each aimed squarely at that clustering-on-energy failure.

Testing a family of variants against one fixed test split inflates the family-wise error, so this module
preregisters, IN CODE and sealed to ``proof/STARSS23_ESCS_BED_VARIANTS.prereg.json`` BEFORE any variant
reads a test score:

- the metric and direction (onset F1 at the DCASE plus-or-minus 200 ms collar, greedy one-to-one, strict
  point-wise PR; direction variant greater than rate-matched-random), restated from the committed prereg;
- the SESOI (0.05), on the identical cost-benefit basis as the committed prereg, recomputed from the
  cached corpus label structure (never from a test score);
- the exact one-sided sign-flip plan (five paired seeds, min one-sided p 1/32, two-sided 0.05 unreachable
  at n equals 5);
- a multiplicity control across the four variants (Bonferroni), which at n equals 5 has a floor above the
  family-adjusted alpha, so no single variant can clear family-wise significance from this family alone:
  a preregistered wall, not a moved goalpost;
- the claim ceiling (the clip is the experimental unit, the verb is bounded to "consistent with");
- the four variant ids, each with a one-line falsifiable hypothesis.

Nothing here reads a test-split F1. The cost-benefit derivation uses only fixed compute anchors and label
structure known before scoring (test-clip count, pooled test onset count, train-set onset density). The
timestamp is passed by the caller, never read from the wall clock inside the sealed body, and the sealed
body carries ``activation_allowed=false`` and ``scientific_promotion=false``.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID, CLAIM_SCOPE
from .prereg import (
    DEFAULT_C_DOWN_FLOPS,
    DEFAULT_C_TRAIN_FLOPS,
    PREREG_DIRECTION,
    PREREG_METRIC,
    PREREGISTERED_SESOI_F1,
    _sesoi_rationale,
    compute_cost_benefit,
)
from .real_artifact import RealBedConfig
from .schema import COLLAR_FRAMES, FRAME_MS
from .stats import BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS

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
    """Raised when the variant preregistration inputs are malformed."""


def _multiplicity_block(n_variants: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:
    """Bonferroni family-wise control across the variant family, honest about the n equals 5 floor."""

    per_variant_alpha = alpha / n_variants
    return {
        "n_variants": n_variants,
        "correction": "Bonferroni",
        "family_alpha": alpha,
        "per_variant_alpha": round(per_variant_alpha, 12),
        "min_achievable_one_sided_p": round(min_one_sided_p, 12),
        "family_significance_reachable_at_n5": bool(min_one_sided_p <= per_variant_alpha),
        "rationale": (
            "four variants scored against one fixed test split inflate the family-wise error, so each "
            f"variant is held to the Bonferroni-adjusted alpha {per_variant_alpha:.4f}. With five paired "
            f"seeds the smallest achievable one-sided sign-flip p is {min_one_sided_p:.5f}, which exceeds "
            "that adjusted alpha, so no single variant can clear family-wise significance from this "
            "family alone. This is a preregistered statistical wall: a variant may still register a "
            "SESOI-exceeding effect for triangulation, but family-wise promotion needs more seeds and "
            "bias-independent reproductions, never a larger claim squeezed from n equals 5"
        ),
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
    """Assemble the self-sealed variant preregistration body. The timestamp is passed, never a clock read."""

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

    cb = compute_cost_benefit(
        c_train_flops=c_train_flops,
        c_down_flops=c_down_flops,
        operating_firing_fraction=operating_firing_fraction,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
    )
    permutations = 2**n_seeds
    min_one_sided_p = 1.0 / permutations

    body: dict[str, Any] = {
        "schema": VARIANTS_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "E1 gate-variant iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
        "metric": PREREG_METRIC,
        "collar_frames": COLLAR_FRAMES,
        "collar_ms": COLLAR_FRAMES * FRAME_MS,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_f1": round(float(sesoi_f1), 12),
            "provisional": False,
            "selection_method": "cost-benefit (docs/ESCS_DEEP_RESEARCH.md Q5a option 3), same as base prereg",
            "rationale": _sesoi_rationale(cb, float(sesoi_f1)),
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
            "statistic": "mean of paired per-seed F1 deltas (variant minus rate_matched_random)",
            "min_one_sided_p": round(min_one_sided_p, 12),
            "two_sided_floor": round(2.0 / permutations, 12),
            "alpha": 0.05,
            "two_sided_alpha_reachable": (2.0 / permutations) <= 0.05,
            "phipson_smyth_applied": False,
        },
        "multiplicity": _multiplicity_block(len(variants), min_one_sided_p, alpha=0.05),
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


def write_variants_prereg(
    body: dict[str, Any], out_path: str | Path = DEFAULT_VARIANTS_PREREG_PATH
) -> Path:
    """Write the self-sealed variant preregistration as canonical JSON bytes for a stable on-disk digest."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path


@dataclass(frozen=True, slots=True)
class StructuralFacts:
    """Label-only structural facts the cost-benefit SESOI needs. Never derived from a test score."""

    operating_firing_fraction: float
    n_test_clips: int
    n_test_onsets: int
    train_onset_density: float
    n_test_frames: int


def structural_facts_from_cache(
    *,
    cache_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> StructuralFacts:
    """Derive the label-only structural facts from the built feature cache. Reads no test-split F1.

    The operating firing fraction is the swept target rate closest to the train-set onset density, the
    exact rule the real producer preregisters, using train labels only.
    """

    from .feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus

    corpus = load_cached_corpus(cache_root=cache_root or DEFAULT_CACHE_ROOT)
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


def _read_base_prereg_digest() -> str | None:
    """Return the committed base prereg canonical digest for traceability, or None if absent."""

    import json

    base_path = Path("proof/STARSS23_ESCS_BED.prereg.json")
    if not base_path.is_file():
        return None
    try:
        base = json.loads(base_path.read_bytes().decode("utf-8"))
    except (ValueError, OSError):
        return None
    digest = base.get("canonical_sha256")
    return str(digest) if isinstance(digest, str) else None


def _main(argv: list[str] | None = None) -> int:
    """Seal the variant preregistration from the cached corpus structural facts and print its digest."""

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
        operating_firing_fraction=facts.operating_firing_fraction,
        n_test_clips=facts.n_test_clips,
        n_test_onsets=facts.n_test_onsets,
        train_onset_density=facts.train_onset_density,
        n_test_frames=facts.n_test_frames,
        base_prereg_canonical_sha256=_read_base_prereg_digest(),
    )
    path = write_variants_prereg(body, args.out)
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
