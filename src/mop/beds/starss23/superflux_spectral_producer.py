
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mop.science import ArtifactResult

from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_TEST_FRAMES,
    PRIMARY_CONTROL,
)
from .feature_cache import load_cached_corpus, load_or_build_cached_corpus
from .featurizer_superflux_spectral import FLOPS_PER_FRAME as SUPERFLUX_FLOPS_PER_FRAME
from .featurizer_superflux_spectral import SuperfluxSpectralFeaturizer
from .featurizer_variant_producer import (
    FeaturizerVariantSpec,
    VariantContext,
    VariantCorpus,
    build_featurizer_variant_artifact,
    featurizer_spread_diagnostic,
)
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealBedConfig,
)
from .schema import COLLAR_FRAMES
from .superflux_spectral_prereg import (
    DEFAULT_FEATURIZERS_PREREG_PATH,
    FEATURIZER_VARIANTS,
    FEATURIZERS_PREREG_SCHEMA,
)

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-superflux-spectral/v1"
VARIANT_ID = "superflux_spectral"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_superflux_spectral.json")

SUPERFLUX_FULL_SCALE_FEATURIZE = SUPERFLUX_FLOPS_PER_FRAME * FULL_SCALE_TEST_FRAMES


class SuperfluxSpectralRefusal(ValueError):
    pass


def _variant_hypothesis() -> str:
    for entry in FEATURIZER_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise SuperfluxSpectralRefusal(f"featurizer {VARIANT_ID!r} is not in the sealed featurizer family")

def build_superflux_spectral_artifact(
    *,
    timestamp: str,
    corpus: Any | None = None,
    cache_root: str | Path | None = None,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> ArtifactResult:

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    if corpus is None:
        if cache_root is None:
            corpus = load_or_build_cached_corpus(front_end="superflux")
        else:
            corpus = load_cached_corpus(front_end="superflux", cache_root=cache_root)
    prepared = VariantCorpus(
        split=corpus.split,
        features_by_clip=corpus.features_by_clip,
        train_density=corpus.train_onset_density(),
        n_test_clips=corpus.n_test_clips(),
        n_test_onsets=corpus.n_test_onsets(),
        n_test_frames=corpus.n_test_frames(),
    )
    featurizer = SuperfluxSpectralFeaturizer()

    def featurizer_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "variant_id": VARIANT_ID,
            "front_end": "superflux_spectral",
            "hypothesis": _variant_hypothesis(),
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": SUPERFLUX_FLOPS_PER_FRAME,
            "base_frontend_flops_per_frame": 1_121_340,
            "feature_dim": 256,
            "feature_cache_key": corpus.cache_key,
            "only_frozen_front_end_differs": True,
            "gate_unchanged": True,
            "no_projection_or_truncation": (
                "the SuperFlux front-end emits exactly 64 mel-flux bins by 4 channels = 256 features, the "
                "gate's hard-wired D_FEAT, so the unchanged gate consumes it with no adaptation and no "
                "trained parameter added"
            ),
            "note": (
                "featurized once with the SuperFlux front-end and cached; the FLOP ledger charges its own "
                "per-frame cost per arm from the cache count, so the cache is a wall-clock optimization "
                "and not a budget cut"
            ),
        }

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "variant_id": VARIANT_ID,
            "variant_kind": "frozen_featurizer",
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": context.beats_random,
            "fire_spread_diagnostic": context.spread,
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": SUPERFLUX_FULL_SCALE_FEATURIZE,
                "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing
                ),
            },
            "real_corpus": {
                "producer_schema": REAL_PRODUCER_SCHEMA,
                "variant_producer_schema": VARIANT_ARTIFACT_SCHEMA,
                "foa_root": str(Path(DEFAULT_FOA_ROOT)),
                "metadata_root": str(Path(DEFAULT_METADATA_ROOT)),
                "feature_cache_key": corpus.cache_key,
                "n_clips": len(corpus.clips),
                "split_rooms": dict(corpus.split.detail),
                "n_train_frames": context.seed_runs[0].train_frames,
                "n_test_clips": prepared.n_test_clips,
                "n_test_onsets": prepared.n_test_onsets,
                "n_test_frames": prepared.n_test_frames,
                "train_onset_density": round(float(prepared.train_density), 12),
                "operating_firing_fraction": round(float(context.operating_rate), 12),
            },
            "prereg": {
                "path": str(Path(featurizers_prereg_path)),
                "canonical_sha256": context.prereg_digest,
                "sesoi_f1": context.sesoi_f1,
                "provisional": False,
                "written_before_test_scores": True,
                "rebuilt_by_this_producer": False,
            },
        }

    spec = FeaturizerVariantSpec(
        artifact_schema=VARIANT_ARTIFACT_SCHEMA,
        variant_id=VARIANT_ID,
        identity_key="variant_id",
        prereg_schema=FEATURIZERS_PREREG_SCHEMA,
        prereg_family_field="variants",
        prereg_member_field="variant_id",
        refusal=SuperfluxSpectralRefusal,
        flops_per_frame=SUPERFLUX_FLOPS_PER_FRAME,
        spread=lambda per_seed: featurizer_spread_diagnostic(
            per_seed,
            definition=(
                "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of "
                "another fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one "
                "referee true positives at the operating budget. Both are computed from the SuperFlux "
                "run's own operating-point fires."
            ),
            anchor_key="committed_null_base_frontend_seed0_anchor",
            source=(
                "docs/mixture_of_perspectives/26_escs_starss23_bed.md "
                "(base half-wave-rectified flux)"
            ),
        ),
        featurizer_payload=featurizer_payload,
        extra_payload=extra_payload,
        final_extra=lambda context: {
            "candidate_featurize_flops": (
                SUPERFLUX_FLOPS_PER_FRAME * context.seed_runs[0].total_frames
            ),
            "candidate_max_lifecycle_flops": max(
                point.candidate.max_lifecycle_flops() for point in context.budget_points
            ),
        },
        receipt_extra={"variant_kind": "frozen_featurizer"},
    )
    return build_featurizer_variant_artifact(
        config=config,
        bed_config=bed_config,
        corpus=prepared,
        featurizer=featurizer,
        prereg_path=featurizers_prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )
