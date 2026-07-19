
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mop.science import ArtifactResult
from mop.science.budget import (
    FlopModel,
    arm_flop_model,
)

from . import FLOP_CEILING
from .artifact import (
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    PRIMARY_CONTROL,
)
from .feature_cache import CachedCorpus, load_or_build_cached_corpus
from .featurizer_spatial_doa import D_FEAT, FLOPS_PER_FRAME, SpatialDoaFeaturizer
from .featurizer_variant_producer import (
    FeaturizerVariantSpec,
    VariantContext,
    VariantCorpus,
    build_featurizer_variant_artifact,
    featurizer_spread_diagnostic,
)
from .gate import FLOPS_PER_INFERENCE, training_flops
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealBedConfig,
)
from .schema import COLLAR_FRAMES
from .spatial_doa_prereg import DEFAULT_FEATURIZERS_PREREG_PATH, FEATURIZERS, FEATURIZERS_PREREG_SCHEMA

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-spatial-doa/v1"
FEATURIZER_ID = "spatial_doa"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_spatial_doa.json")

FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME * FULL_SCALE_TEST_FRAMES


class SpatialDoaRefusal(ValueError):
    pass


def _featurizer_hypothesis() -> str:
    for entry in FEATURIZERS:
        if entry["featurizer_id"] == FEATURIZER_ID:
            return entry["hypothesis"]
    raise SpatialDoaRefusal(f"featurizer {FEATURIZER_ID!r} is not in the sealed featurizer family")




def _flop_model(kind: str, total_frames: int, train_frames: int, epochs: int) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME,
        gate_infer_per_frame=FLOPS_PER_INFERENCE,
        downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        candidate_train_flops=lambda: training_flops(train_frames, epochs),
    )

def build_spatial_doa_artifact(
    *,
    timestamp: str,
    corpus: CachedCorpus | None = None,
    cache_root: str | Path | None = None,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> ArtifactResult:

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    if corpus is None:
        kwargs: dict[str, Any] = {} if cache_root is None else {"cache_root": cache_root}
        corpus = load_or_build_cached_corpus(front_end="spatial_doa", **kwargs)
    prepared = VariantCorpus(
        split=corpus.split,
        features_by_clip=corpus.features_by_clip,
        train_density=corpus.train_onset_density(),
        n_test_clips=corpus.n_test_clips(),
        n_test_onsets=corpus.n_test_onsets(),
        n_test_frames=corpus.n_test_frames(),
    )
    featurizer = SpatialDoaFeaturizer()

    def featurizer_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "front_end": "spatial_doa_active_intensity",
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "d_feat": D_FEAT,
            "flops_per_frame": FLOPS_PER_FRAME,
            "feature_cache_key": corpus.cache_key,
            "note": (
                "frozen zero-trained-parameter active-intensity DOA front-end (64 bands x [dir_x, dir_y, "
                "dir_z, diffuseness] = 256); featurized once and cached; the FLOP ledger charges it per "
                "arm from the honest per-frame count, so caching is not a budget cut"
            ),
        }

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "featurizer_id": FEATURIZER_ID,
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": context.beats_random,
            "featurizer_swap": {
                "featurizer_id": FEATURIZER_ID,
                "hypothesis": _featurizer_hypothesis(),
                "front_end": (
                    "per-band active-intensity direction of arrival (az/el direction cosines) + "
                    "DirAC diffuseness"
                ),
                "replaces": "half-wave-rectified log-mel spectral flux",
                "only_front_end_differs": True,
                "gate_unchanged": True,
                "output_dim": D_FEAT,
                "output_dim_note": (
                    "emits exactly 256 features natively, so the unchanged gate consumes it with no "
                    "projection and no truncation; the gate's hardcoded length-256 feature contract holds"
                ),
                "featurizers_prereg_path": str(Path(featurizers_prereg_path)),
                "featurizers_prereg_canonical_sha256": context.prereg_digest,
                "fire_spread_diagnostic": context.spread,
            },
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing
                ),
            },
            "real_corpus": {
                "producer_schema": REAL_PRODUCER_SCHEMA,
                "featurizer_producer_schema": VARIANT_ARTIFACT_SCHEMA,
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
        variant_id=FEATURIZER_ID,
        identity_key="featurizer_id",
        prereg_schema=FEATURIZERS_PREREG_SCHEMA,
        prereg_family_field="featurizers",
        prereg_member_field="featurizer_id",
        refusal=SpatialDoaRefusal,
        flops_per_frame=FLOPS_PER_FRAME,
        spread=lambda per_seed: featurizer_spread_diagnostic(
            per_seed,
            definition=(
                "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of "
                "another fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one "
                "referee true positives at the operating budget, under the spatial-DOA front-end"
            ),
            anchor_key="committed_null_seed0_anchor",
            source=(
                "docs/mixture_of_perspectives/26_escs_starss23_bed.md "
                "(log-mel flux front-end)"
            ),
        ),
        featurizer_payload=featurizer_payload,
        extra_payload=extra_payload,
        final_extra=lambda context: {
            "featurizer_flops_per_frame": FLOPS_PER_FRAME,
            "candidate_max_lifecycle_flops": int(context.report.matched_budget.flops),
            "flop_ceiling": FLOP_CEILING,
        },
        receipt_extra={},
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
