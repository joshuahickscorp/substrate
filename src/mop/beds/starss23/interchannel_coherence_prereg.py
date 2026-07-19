
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

DEFAULT_FEATURIZERS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_interchannel_coherence.prereg.json")

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
    pass


_FAMILY_PREREG = {
    "schema": FEATURIZERS_PREREG_SCHEMA,
    "wave": "F1 featurizer-swap iteration",
    "members_field": "variants",
    "member_id_field": "variant_id",
    "member_label": "featurizer",
    "family_phrase": "three featurizers",
    "n_field": "n_variants",
    "per_alpha_field": "per_variant_alpha",
    "alpha_digits": 4,
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
    "malformed_message": "each featurizer needs a variant_id and a one-line hypothesis",
    "extra": {
        "featurizer_contract": (
            "each preregistered featurizer is a frozen zero-trained-parameter front-end emitting exactly "
            "256 features per 100 ms frame, scored through the unchanged sealed gate (264 inputs, 3193 "
            "trainable parameters); the featurizer's per-frame FLOPs are charged identically to every arm "
            "and every arm total stays under the 6e10 lifecycle ceiling"
        )
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


def structural_facts_from_adapter(
    *,
    foa_root: str | Path | None = None,
    metadata_root: str | Path | None = None,
    target_rates: tuple[float, ...] | None = None,
    n_val_rooms: int | None = None,
) -> StructuralFacts:

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
