
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

FEATURIZERS_PREREG_SCHEMA = "mop-starss23-escs-bed-featurizers-prereg/v1"
STAGE = 3
N_PAIRED_SEEDS = 5

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_spatial_doa.prereg.json")

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


_FAMILY_PREREG = {
    "schema": FEATURIZERS_PREREG_SCHEMA,
    "wave": "F1 frozen-featurizer iteration",
    "members_field": "featurizers",
    "member_id_field": "featurizer_id",
    "member_label": "featurizer",
    "family_phrase": "three frozen featurizers",
    "n_field": "n_featurizers",
    "per_alpha_field": "per_featurizer_alpha",
    "alpha_digits": 6,
    "statistic": "mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
    "promotion_bar": (
        "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
        "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
        "same direction; a single run, and any run at n equals 5 across this three-featurizer family, "
        "can never promote"
    ),
    "refusal": FeaturizersPreregRefusal,
    "empty_message": "at least one featurizer must be preregistered",
    "duplicate_message": "featurizer ids must be unique",
    "malformed_message": "each featurizer needs a featurizer_id and a one-line hypothesis",
    "extra": {
        "front_end_swap": (
            "each family replaces the frozen zero-trained-parameter featurizer; the trained gate, its "
            "parameter count and ceiling, the sealed referee, controls, harness, and sign-flip are "
            "unchanged, so any F1 difference is attributable to the front-end, not the model"
        ),
        "executed_in_this_wave": ["spatial_doa"],
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
    featurizers: tuple[dict[str, str], ...] = FEATURIZERS,
    base_prereg_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    options = locals()
    options["members"] = options.pop("featurizers")
    return build_family_prereg(**options, **_FAMILY_PREREG)




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
