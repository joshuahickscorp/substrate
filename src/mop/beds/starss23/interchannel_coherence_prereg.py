"""Sealed preregistration for the F1 featurizer-swap iteration wave (three frozen featurizer families).

This is a net-new, additive component. It changes no sealed scoring logic. The committed gate-variant
wave held the frozen log-mel spectral-flux front-end fixed and changed only the firing policy; every arm
nulled because a per-channel ENERGY front-end gives the gate no cue that separates a direct-sound onset
from loud steady ambience. This wave instead swaps the FROZEN featurizer for a new zero-trained-parameter
front-end and scores it through the SAME sealed gate, harness, referee, and controls.

Testing a family of featurizers against one fixed test split inflates the family-wise error, so this
module preregisters, IN CODE and sealed BEFORE any featurizer reads a test score:

- the metric and direction (onset F1 at the DCASE plus-or-minus 200 ms collar, greedy one-to-one, strict
  point-wise PR; direction candidate greater than rate-matched-random), restated from the committed prereg;
- the SESOI (0.05), on the identical cost-benefit basis as the committed prereg, recomputed from the
  corpus label structure (never from a test score);
- the exact one-sided sign-flip plan (five paired seeds, min one-sided p 1/32, two-sided 0.05 unreachable
  at n equals 5);
- a multiplicity control across the THREE featurizer families (Bonferroni), whose per-family adjusted
  alpha 0.05/3 = 0.016667 sits BELOW the smallest achievable one-sided sign-flip p 1/32 = 0.03125, so no
  single featurizer can clear family-wise significance from this family alone: a preregistered wall, not a
  moved goalpost;
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
    bonferroni_family,
    family_analysis_plan,
    family_cli_summary,
)
from .real_artifact import RealBedConfig

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_interchannel_coherence.prereg.json")

# The three F1 featurizer families. Each is a distinct, falsifiable frozen zero-trained-parameter
# front-end aimed at the same diagnosed gap: a per-channel ENERGY front-end cannot tell a direct-sound
# onset from loud steady ambience. Each front-end emits exactly 256 features so the unchanged sealed gate
# consumes it, and each is scored against the same sealed referee and rate-matched-random control. Only
# ``interchannel_coherence`` is built and run in this wave; all three define the multiplicity family.
FEATURIZER_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "interchannel_coherence",
        "hypothesis": (
            "Replacing per-channel log-mel energy flux with the frozen magnitude-squared coherence "
            "between the FOA omni channel W and each gradient channel X, Y, Z per band, plus the DirAC "
            "directness (one minus diffuseness) per band, gives the gate a direct-versus-diffuse cue "
            "orthogonal to energy: a transient onset is a coherent, low-diffuseness direct sound against "
            "diffuse ambience, so the gate places its fixed budget on direct-sound onsets and beats free "
            "random placement at matched budget."
        ),
    },
    {
        "variant_id": "spatial_doa",
        "hypothesis": (
            "Replacing energy flux with the frozen per-band DirAC direction-of-arrival (azimuth and "
            "elevation) and its short-horizon angular dispersion gives the gate a direction-stability "
            "cue: an onset introduces a new, momentarily stable arrival direction against the "
            "directionally scrambled diffuse field, so the gate separates onsets from ambience and beats "
            "free random placement."
        ),
    },
    {
        "variant_id": "modulation_flux",
        "hypothesis": (
            "Replacing raw flux with the frozen per-band amplitude-modulation-spectrum energy (a short "
            "temporal-modulation front-end over the log-mel envelope) gives the gate a modulation cue: "
            "onsets carry broadband high-rate temporal modulation absent from steady tones and "
            "stationary noise, so the gate sharpens onset placement over free random placement."
        ),
    },
)


class FeaturizersPreregRefusal(ValueError):
    """Raised when the featurizer preregistration inputs are malformed."""


def _multiplicity_block(n_variants: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:
    """Bonferroni family-wise control across the featurizer family, honest about the n equals 5 floor."""

    return bonferroni_family(
        n_variants, min_one_sided_p, alpha,
        family_phrase="three featurizers", member_label="featurizer",
        n_field="n_variants", per_alpha_field="per_variant_alpha", alpha_digits=4,
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
    """Assemble the self-sealed featurizer preregistration body.

    The timestamp is passed, never a clock read.
    """

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise FeaturizersPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise FeaturizersPreregRefusal("n_seeds must be positive")
    if not variants:
        raise FeaturizersPreregRefusal("at least one featurizer must be preregistered")
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
        "wave": "F1 featurizer-swap iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
        "featurizer_contract": (
            "each preregistered featurizer is a frozen zero-trained-parameter front-end emitting exactly "
            "256 features per 100 ms frame, scored through the unchanged sealed gate (264 inputs, 3193 "
            "trainable parameters); the featurizer's per-frame FLOPs are charged identically to every arm "
            "and every arm total stays under the 6e10 lifecycle ceiling"
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


def structural_facts_from_adapter(
    *,
    foa_root: str | Path | None = None,
    metadata_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
    n_val_rooms: int | None = None,
) -> StructuralFacts:
    """Derive the label-only structural facts from a fresh real-adapter split. Reads no test-split F1.

    Only clip onset counts and frame counts are read (labels), never a featurized frame or a score. The
    operating firing fraction is the swept target rate closest to the train-set onset density, the exact
    rule the producer preregisters, using train labels only.
    """

    from .adapter import RealStarssAdapter, native_fold_split
    from .real_artifact import (
        DEFAULT_FOA_ROOT,
        DEFAULT_METADATA_ROOT,
        DEFAULT_N_VAL_ROOMS,
        RealArtifactRefusal,
    )

    config = RealBedConfig()
    rates = target_rates or config.target_rates
    adapter = RealStarssAdapter(
        foa_root or DEFAULT_FOA_ROOT,
        metadata_root or DEFAULT_METADATA_ROOT,
        rights_clean=True,
        max_frames=None,
    )
    split = native_fold_split(
        adapter,
        n_val_rooms if n_val_rooms is not None else DEFAULT_N_VAL_ROOMS,
        refusal=RealArtifactRefusal,
        refuse_empty=False,
    )
    return StructuralFacts.from_split(split, rates)


def _main(argv: list[str] | None = None) -> int:
    """Seal the featurizer preregistration from the real-adapter structural facts and print its digest."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Seal the STARSS23 ESCS interchannel-coherence featurizer-family preregistration."
    )
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    parser.add_argument("--foa-root", default=None)
    parser.add_argument("--metadata-root", default=None)
    parser.add_argument("--out", default=str(DEFAULT_FEATURIZERS_PREREG_PATH))
    args = parser.parse_args(argv)

    facts = structural_facts_from_adapter(foa_root=args.foa_root, metadata_root=args.metadata_root)
    body = build_featurizers_prereg(
        timestamp=args.timestamp,
        **facts.payload(),
        base_prereg_canonical_sha256=base_prereg_digest(),
    )
    path = write_canonical_json(body, args.out)
    print(f"wrote {path}")
    print(
        json.dumps(
            family_cli_summary(
                body, "n_variants", "variants", "variant_id", "variant_ids",
                ("per_variant_alpha", "min_achievable_one_sided_p"),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
