
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
)
from mop.science.gating import causal_gate_trace
from mop.substrate.events import write_canonical_json

from .adapter import (
    RealStarssAdapter,
    domain_seed,
    map_clip_audio,
    marginal_matched_noise,
    native_fold_split,
)
from .adapter import (
    onset_density as _onset_density,
)
from .artifact import (
    ARTIFACT_SCHEMA,
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    BedConfig,
    _pooled_score,
    _SeedRun,
    _train_gate,
)
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate import OnlineState
from .prereg import DEFAULT_PREREG_PATH, build_prereg
from .referee import score_arm
from .schema import COLLAR_FRAMES, ClipSplit

REAL_PRODUCER_SCHEMA = "mop-starss23-escs-real-producer/v1"

# The real STARSS23 FOA subset and metadata roots on this host.
DEFAULT_FOA_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
DEFAULT_METADATA_ROOT = Path(
    "/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev"
)

# Val rooms are carved from the fold-3 dev-train rooms so train, val, and test stay room-disjoint and the
# test partition is exactly the held-out fold-4 dev-test.
DEFAULT_N_VAL_ROOMS = 2


class RealArtifactRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RealBedConfig:

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS
    target_rates: tuple[float, ...] = (0.10, 0.08, 0.06)
    noisy_tv_frames: int = 2000
    max_frames: int | None = None

    def bed_config(self) -> BedConfig:

        return BedConfig(
            seeds=self.seeds,
            target_rates=self.target_rates,
            noisy_tv_frames=self.noisy_tv_frames,
            downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        )


# ---------------------------------------------------------------------------
# The real noisy-TV channel: white-noise audio featurized and marginal-matched to the real test content.
# ---------------------------------------------------------------------------


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: FrozenFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:

    noise_seed = domain_seed(
        seed, "mop.beds.starss23.real.noisy_tv", b"mop-starss23-real-noisy-tv-v1"
    )
    return marginal_matched_noise(noise_seed, n_frames, featurizer, target_mean, target_std)


# ---------------------------------------------------------------------------
# Per-seed run on the fixed real split.
# ---------------------------------------------------------------------------


def _run_seed_real(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: BedConfig,
    operating_density: float,
) -> _SeedRun:

    gate, train_frames = _train_gate(seed, split.train, features_by_clip, config)
    total_frames = int(sum(clip.n_frames for clip in split.test))

    val_probs = np.concatenate(
        [causal_gate_trace(gate, features_by_clip[clip.clip_id], 0.5, OnlineState.initial)[1]
         for clip in split.val]
    )
    best_single = BestSingleControl.tuned(
        [(features_by_clip[clip.clip_id], list(clip.onset_frames)) for clip in split.val]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    for rate in config.target_rates:
        theta = float(np.quantile(val_probs, 1.0 - rate))
        budget_id = f"rate_{rate:.2f}"
        val_scored = []
        for clip in split.val:
            fires, _ = causal_gate_trace(
                gate, features_by_clip[clip.clip_id], theta, OnlineState.initial
            )
            val_scored.append((list(clip.onset_frames), fires))
        val_f1 = score_arm(val_scored, COLLAR_FRAMES).f1

        clips_block: list[dict[str, Any]] = []
        arm_clip_scores: dict[str, list[tuple[list[int], list[int]]]] = {
            ARM_CANDIDATE: [],
            ARM_RATE_MATCHED_RANDOM: [],
            ARM_ALWAYS_ON: [],
            ARM_BEST_SINGLE: [],
        }
        firings = {kind: 0 for kind in arm_clip_scores}
        for clip in split.test:
            features = features_by_clip[clip.clip_id]
            gt = list(clip.onset_frames)
            candidate_fires, _ = causal_gate_trace(
                gate, features, theta, OnlineState.initial
            )
            fires = {
                ARM_CANDIDATE: candidate_fires,
                ARM_RATE_MATCHED_RANDOM: rate_matched_random_fires(
                    candidate_fires, clip.n_frames, seed=seed, clip_id=clip.clip_id
                ),
                ARM_ALWAYS_ON: always_on_fires(clip.n_frames),
                ARM_BEST_SINGLE: best_single.fires_for_clip(features),
            }
            for kind, arm_fires in fires.items():
                arm_clip_scores[kind].append((gt, arm_fires))
                firings[kind] += len(arm_fires)
            clips_block.append(
                {
                    "clip_id": clip.clip_id,
                    "gt_onsets": gt,
                    "fires": {kind: list(arm_fires) for kind, arm_fires in fires.items()},
                }
            )
        arm_scores = {kind: _pooled_score(pairs) for kind, pairs in arm_clip_scores.items()}
        per_budget[budget_id] = {
            "theta": theta,
            "rate": rate,
            "val_f1": val_f1,
            "clips": clips_block,
            "arm_scores": arm_scores,
            "firings": firings,
        }

    # Preregistered operating point: the swept firing budget whose rate is closest to the train-set onset
    # density. A fixed rule set before scoring, using only train labels, never a val or test F1 argmax.
    operating_budget_id = min(
        per_budget, key=lambda bid: abs(per_budget[bid]["rate"] - operating_density)
    )
    operating = per_budget[operating_budget_id]
    per_seed_block = {
        "seed": seed,
        "operating_budget_id": operating_budget_id,
        "clips": operating["clips"],
        "arm_scores": operating["arm_scores"],
    }

    operating_theta = operating["theta"]
    base_rate = operating["firings"][ARM_CANDIDATE] / max(1, total_frames)
    noise_fires, _ = causal_gate_trace(
        gate, noise_features, operating_theta, OnlineState.initial
    )
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    return _SeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
    )


# ---------------------------------------------------------------------------
# Assemble and seal the real artifact.
# ---------------------------------------------------------------------------


def build_real_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_PREREG_PATH,
) -> ArtifactResult:

    from .featurizer_variant_producer import (
        FeaturizerVariantSpec,
        VariantContext,
        VariantCorpus,
        build_featurizer_variant_artifact,
    )

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = FrozenFeaturizer()
    adapter = RealStarssAdapter(
        foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames
    )
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    split = native_fold_split(
        adapter,
        config.n_val_rooms,
        refusal=RealArtifactRefusal,
        refuse_empty=False,
    )
    prepared = VariantCorpus(
        split=split,
        features_by_clip=features_by_clip,
        train_density=_onset_density(split.train),
        n_test_clips=len(split.test),
        n_test_onsets=sum(len(clip.onsets) for clip in split.test),
        n_test_frames=int(sum(clip.n_frames for clip in split.test)),
    )
    def prepare_prereg(current: VariantCorpus) -> tuple[dict[str, Any], str | Path]:
        prereg = build_prereg(
            timestamp=timestamp,
            operating_firing_fraction=min(
                bed_config.target_rates,
                key=lambda rate: abs(rate - current.train_density),
            ),
            n_test_clips=current.n_test_clips,
            n_test_onsets=current.n_test_onsets,
            train_onset_density=current.train_density,
            n_test_frames=current.n_test_frames,
        )
        return prereg, write_canonical_json(prereg, prereg_path)

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        truncations = [truncation.payload() for truncation in adapter.truncations()]
        return {
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
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
                    "clips_capped_by_max_frames": sum(
                        1 for item in truncations if item["capped_by_max_frames"]
                    ),
                    "onsets_dropped_past_audio_end": sum(
                        item["dropped_onsets_past_end"] for item in truncations
                    ),
                    "max_frames": config.max_frames,
                    "per_clip": truncations,
                },
            },
            "prereg": {
                "path": str(Path(prereg_path)),
                "canonical_sha256": context.prereg_digest,
                "sesoi_f1": context.sesoi_f1,
                "provisional": False,
                "written_before_test_scores": True,
            },
        }

    spec = FeaturizerVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        variant_id="",
        identity_key=None,
        prereg_schema="",
        prereg_family_field="",
        prereg_member_field="",
        refusal=RealArtifactRefusal,
        flops_per_frame=FLOPS_PER_FRAME,
        spread=lambda _per_seed: {},
        featurizer_payload=lambda _context: {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        },
        extra_payload=extra_payload,
        final_extra=lambda _context: {},
        receipt_extra={},
        receipt_note=(
            "one real run is a mechanics demonstration; scientific confirmation needs the independent "
            "verifier plus at least three bias-independent reproductions and cannot be self-certified"
        ),
        prepare_prereg=prepare_prereg,
        include_prereg_in_result=True,
        include_spread_in_detail=False,
        stats_extra=lambda _beats_random: {},
        include_beats_random_in_detail=False,
    )
    return build_featurizer_variant_artifact(
        config=config,
        bed_config=bed_config,
        corpus=prepared,
        featurizer=featurizer,
        prereg_path=prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )
