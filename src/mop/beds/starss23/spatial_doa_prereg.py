
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

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_spatial_doa.prereg.json")

# The three frozen zero-trained-parameter featurizer families. Each is a distinct, falsifiable answer to
# the same diagnosed failure (the log-mel front-end discards the spatial channel, so the gate cannot tell
# two equally loud events apart by their direction). Each hypothesis is one line, scored against the same
# sealed referee and rate-matched-random control. Only ``spatial_doa`` is executed in this wave; the other
# two are preregistered members of the multiplicity family so the family-wise wall is honest.
FEATURIZERS: tuple[dict[str, str], ...] = (
    {
        "featurizer_id": "spatial_doa",
        "hypothesis": (
            "Replacing the log-mel flux front-end with per-band active-intensity direction of arrival "
            "(the source azimuth and elevation as wraparound-free direction cosines) plus a DirAC "
            "diffuseness exposes the direction a new event arrives from, which the gate can use to place "
            "the fixed firing budget on distinct directional onsets and beat rate-matched-random."
        ),
    },
    {
        "featurizer_id": "spatial_coherence",
        "hypothesis": (
            "A frozen per-band interchannel magnitude-squared coherence between the omni and the three "
            "dipole channels flags coherent single-source arrivals against reverberant or diffuse frames, "
            "so the gate withholds fires in incoherent frames and spends the budget on coherent onsets, "
            "beating uniform-random placement at matched budget."
        ),
    },
    {
        "featurizer_id": "doa_flux",
        "hypothesis": (
            "A frozen half-wave-rectified temporal flux of the per-band direction of arrival (the rate of "
            "directional change) fires the explicit new-direction novelty of an onset, so the gate keys on "
            "directional change points rather than steady directions and places its budget better than "
            "rate-matched-random."
        ),
    },
)


class FeaturizersPreregRefusal(ValueError):
    pass


def _multiplicity_block(n_featurizers: int, min_one_sided_p: float, alpha: float) -> dict[str, Any]:

    return bonferroni_family(
        n_featurizers, min_one_sided_p, alpha,
        family_phrase="three frozen featurizers", member_label="featurizer",
        n_field="n_featurizers", per_alpha_field="per_featurizer_alpha", alpha_digits=6,
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
    featurizers: tuple[dict[str, str], ...] = FEATURIZERS,
    base_prereg_canonical_sha256: str | None = None,
) -> dict[str, Any]:

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise FeaturizersPreregRefusal("timestamp must be a non-empty string passed by the caller")
    if n_seeds <= 0:
        raise FeaturizersPreregRefusal("n_seeds must be positive")
    if not featurizers:
        raise FeaturizersPreregRefusal("at least one featurizer must be preregistered")
    ids = [entry["featurizer_id"] for entry in featurizers]
    if len(set(ids)) != len(ids):
        raise FeaturizersPreregRefusal("featurizer ids must be unique")
    for entry in featurizers:
        if not entry.get("featurizer_id") or not entry.get("hypothesis"):
            raise FeaturizersPreregRefusal("each featurizer needs a featurizer_id and a one-line hypothesis")

    body: dict[str, Any] = {
        "schema": FEATURIZERS_PREREG_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "F1 frozen-featurizer iteration",
        "base_prereg_canonical_sha256": base_prereg_canonical_sha256,
        "front_end_swap": (
            "each family replaces the frozen zero-trained-parameter featurizer; the trained gate, its "
            "parameter count and ceiling, the sealed referee, controls, harness, and sign-flip are "
            "unchanged, so any F1 difference is attributable to the front-end, not the model"
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
            n_members=len(featurizers),
            statistic="mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
            multiplicity=_multiplicity_block,
        ),
        "featurizers": [
            {"featurizer_id": entry["featurizer_id"], "hypothesis": entry["hypothesis"]}
            for entry in featurizers
        ],
        "executed_in_this_wave": ["spatial_doa"],
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


# ---------------------------------------------------------------------------
# Label-only structural facts. Read the fold-respecting split's labels only, never a feature or a score.
# ---------------------------------------------------------------------------


def structural_facts_from_adapter(
    *,
    foa_root: str | Path | None = None,
    metadata_root: str | Path | None = None,
    max_frames: int | None = None,
    n_val_rooms: int | None = None,
    target_rates: tuple[float, ...] | None = None,
) -> dict[str, Any]:

    from .adapter import RealStarssAdapter, native_fold_split
    from .real_artifact import (
        DEFAULT_FOA_ROOT,
        DEFAULT_METADATA_ROOT,
        DEFAULT_N_VAL_ROOMS,
        RealArtifactRefusal,
        RealBedConfig,
    )

    foa = Path(foa_root) if foa_root is not None else DEFAULT_FOA_ROOT
    meta = Path(metadata_root) if metadata_root is not None else DEFAULT_METADATA_ROOT
    n_val = DEFAULT_N_VAL_ROOMS if n_val_rooms is None else int(n_val_rooms)
    rates = target_rates or RealBedConfig().target_rates

    adapter = RealStarssAdapter(foa, meta, rights_clean=True, max_frames=max_frames)
    split = native_fold_split(
        adapter, n_val, refusal=RealArtifactRefusal, refuse_empty=False
    )

    return StructuralFacts.from_split(split, rates).payload()


def _main(argv: list[str] | None = None) -> int:

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Seal the STARSS23 ESCS frozen-featurizer preregistration."
    )
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    parser.add_argument("--foa", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default=str(DEFAULT_FEATURIZERS_PREREG_PATH))
    args = parser.parse_args(argv)

    facts = structural_facts_from_adapter(foa_root=args.foa, metadata_root=args.metadata)
    body = build_featurizers_prereg(
        timestamp=args.timestamp,
        **facts,
        base_prereg_canonical_sha256=base_prereg_digest(),
    )
    path = write_canonical_json(body, args.out)
    print(f"wrote {path}")
    print(
        json.dumps(
            family_cli_summary(body, "n_featurizers", "featurizers", "featurizer_id", "featurizer_ids"),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
