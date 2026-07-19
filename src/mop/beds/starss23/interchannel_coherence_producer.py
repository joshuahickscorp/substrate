
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mop.science import ArtifactResult

from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split
from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
)
from .featurizer_interchannel_coherence import (
    D_FEAT,
    FLOPS_PER_FRAME,
    InterchannelCoherenceFeaturizer,
)
from .featurizer_variant_producer import (
    FeaturizerVariantSpec,
    VariantContext,
    VariantCorpus,
    build_featurizer_variant_artifact,
    featurizer_spread_diagnostic,
)
from .interchannel_coherence_prereg import (
    DEFAULT_FEATURIZERS_PREREG_PATH,
    FEATURIZER_VARIANTS,
    FEATURIZERS_PREREG_SCHEMA,
)
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealArtifactRefusal,
    RealBedConfig,
    _onset_density,
)
from .schema import COLLAR_FRAMES

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-interchannel-coherence/v1"
VARIANT_ID = "interchannel_coherence"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_interchannel_coherence.json")


class InterchannelCoherenceRefusal(ValueError):
    pass


def _variant_hypothesis() -> str:
    for entry in FEATURIZER_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise InterchannelCoherenceRefusal(f"featurizer {VARIANT_ID!r} is not in the sealed featurizer family")

def build_interchannel_coherence_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> ArtifactResult:

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = InterchannelCoherenceFeaturizer()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    split = native_fold_split(
        adapter, config.n_val_rooms, refusal=RealArtifactRefusal, refuse_empty=False
    )
    prepared = VariantCorpus(
        split=split,
        features_by_clip=features_by_clip,
        train_density=_onset_density(split.train),
        n_test_clips=len(split.test),
        n_test_onsets=sum(len(clip.onsets) for clip in split.test),
        n_test_frames=int(sum(clip.n_frames for clip in split.test)),
    )
    truncations = [truncation.payload() for truncation in adapter.truncations()]
    dropped_onsets = sum(item["dropped_onsets_past_end"] for item in truncations)
    capped_clips = sum(1 for item in truncations if item["capped_by_max_frames"])

    def featurizer_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "family": VARIANT_ID,
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
            "d_feat": D_FEAT,
            "note": (
                "frozen zero-trained-parameter interchannel-coherence front-end; featurized inline (its "
                "digest differs from the frozen cache) and charged per arm from its honest per-frame count"
            ),
        }

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "variant_id": VARIANT_ID,
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": context.beats_random,
            "variant": {
                "variant_id": VARIANT_ID,
                "kind": "featurizer_swap",
                "hypothesis": _variant_hypothesis(),
                "front_end": (
                    "frozen magnitude-squared coherence between W and X, Y, Z per band plus DirAC "
                    "directness per band; 64 bands by 4 spatial features equals 256, the exact width the "
                    "sealed gate consumes, so the gate and its parameter count are unchanged"
                ),
                "only_featurizer_differs": True,
                "gate_unchanged": True,
                "featurizers_prereg_path": str(Path(featurizers_prereg_path)),
                "featurizers_prereg_canonical_sha256": context.prereg_digest,
                "fire_spread_diagnostic": context.spread,
            },
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "featurize_flops_24000_frames_this_featurizer": FLOPS_PER_FRAME * 24_000,
                "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing
                ),
            },
            "real_corpus": {
                "producer_schema": REAL_PRODUCER_SCHEMA,
                "variant_producer_schema": VARIANT_ARTIFACT_SCHEMA,
                "foa_root": str(Path(foa_root)),
                "metadata_root": str(Path(metadata_root)),
                "n_clips": len(adapter.clips()),
                "split_rooms": split.detail,
                "n_train_frames": context.seed_runs[0].train_frames,
                "n_test_clips": prepared.n_test_clips,
                "n_test_onsets": prepared.n_test_onsets,
                "n_test_frames": prepared.n_test_frames,
                "train_onset_density": round(float(prepared.train_density), 12),
                "operating_firing_fraction": round(float(context.operating_rate), 12),
                "truncation": {
                    "clips_capped_by_max_frames": capped_clips,
                    "onsets_dropped_past_audio_end": dropped_onsets,
                    "max_frames": config.max_frames,
                    "per_clip": truncations,
                },
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
        refusal=InterchannelCoherenceRefusal,
        flops_per_frame=FLOPS_PER_FRAME,
        spread=lambda per_seed: featurizer_spread_diagnostic(
            per_seed,
            definition=(
                "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of "
                "another fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one "
                "referee true positives at the operating budget."
            ),
            anchor_key="committed_null_seed0_anchor",
            source="docs/mixture_of_perspectives/26_escs_starss23_bed.md",
        ),
        featurizer_payload=featurizer_payload,
        extra_payload=extra_payload,
        final_extra=lambda context: {
            "featurizer_flops_per_frame": FLOPS_PER_FRAME,
            "candidate_max_lifecycle_flops": context.report.matched_budget.payload()["flops"],
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
