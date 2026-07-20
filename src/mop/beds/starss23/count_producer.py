from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.config import REPO_ROOT
from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    BudgetSeedRun,
    FlopModel,
    arm_flop_model,
)
from mop.science.gating import assemble_causal_inputs, causal_gate_trace

from .adapter import RealStarssAdapter, domain_seed, marginal_matched_noise, native_fold_split
from .controls import (
    always_on_fires,
    at_chance,
    never_update_reestimates,
    rate_matched_random_fires,
)
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import (
    COUNT_VOC_WINDOW,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PONDER_LAMBDA,
    FLOPS_PER_INFERENCE,
    CountGate,
    CountOnlineState,
    training_flops,
    voc_targets_from_count_track,
)
from .count_prereg import (
    DEFAULT_COUNT_PREREG_PATH,
    build_count_prereg,
)
from .count_referee import COLD_START, score_arm
from .count_variant_producer import (
    CountVariantContext,
    CountVariantSpec,
    build_count_variant_artifact,
    prepare_count_variant_corpus,
)
from .schema import Clip

COUNT_PRODUCER_SCHEMA = "mop-starss23-count-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

DEFAULT_STARSS_ROOT = Path(os.environ.get("MOP_STARSS23_ROOT", REPO_ROOT / "data" / "starss23"))
DEFAULT_FOA_ROOT = DEFAULT_STARSS_ROOT / "foa_subset" / "foa_dev"
DEFAULT_METADATA_ROOT = DEFAULT_STARSS_ROOT / "metadata_dev_extracted" / "metadata_dev"

DEFAULT_N_VAL_ROOMS = 2

FULL_SCALE_TRAIN_FRAMES = 54_000
FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_C_TRAIN = training_flops(FULL_SCALE_TRAIN_FRAMES, DEFAULT_EPOCHS)  # ~8.27e9
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME_COUNT * FULL_SCALE_TEST_FRAMES


class CountProducerRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RealCountBedConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS
    target_rates: tuple[float, ...] = (0.10, 0.05, 0.02)
    noisy_tv_frames: int = 2000
    max_frames: int | None = None
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    ponder_lambda: float = DEFAULT_PONDER_LAMBDA
    voc_window: int = COUNT_VOC_WINDOW
    downstream_flops_per_reestimate: int = FLOPS_PER_REESTIMATE

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise CountProducerRefusal("the bed needs at least two paired seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise CountProducerRefusal("paired seeds must be unique")
        if not self.target_rates:
            raise CountProducerRefusal("at least one re-estimation budget target rate is required")


def _train_count_gate(
    seed: int,
    train_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    config: RealCountBedConfig,
) -> tuple[CountGate, int]:

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, CountOnlineState.initial))
        targets.append(voc_targets_from_count_track(gt_by_clip[clip.clip_id], window=config.voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = CountGate(seed=seed)
    gate.fit(
        x,
        y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
    )
    return gate, int(x.shape[0])


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: FrozenCountFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:

    noise_seed = domain_seed(seed, "mop.beds.starss23.count.noisy_tv", b"mop-starss23-count-noisy-tv-v1")
    return marginal_matched_noise(noise_seed, n_frames, featurizer, target_mean, target_std)


def run_count_seed(
    *,
    seed: int,
    val_clips: tuple[Clip, ...],
    test_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    estimator_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    noise_features: np.ndarray,
    target_rates: tuple[float, ...],
    operating_density: float,
    train_gate: Callable[[], tuple[Any, int]],
    state_factory: Callable[[], Any],
    score_rows: Callable[[list[tuple[str, list[int], list[int], list[int]]]], dict[str, Any]],
) -> BudgetSeedRun:

    gate, train_frames = train_gate()
    total_frames = int(sum(clip.n_frames for clip in test_clips))

    val_probs = np.concatenate(
        [causal_gate_trace(gate, features_by_clip[clip.clip_id], 0.5, state_factory)[1] for clip in val_clips]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    for rate in target_rates:
        theta = float(np.quantile(val_probs, 1.0 - rate))
        budget_id = f"rate_{rate:.2f}"

        arm_rows: dict[str, list[tuple[str, list[int], list[int], list[int]]]] = {
            ARM_CANDIDATE: [],
            ARM_RATE_MATCHED_RANDOM: [],
            ARM_ALWAYS_ON: [],
            ARM_NEVER_UPDATE: [],
        }
        reestimations = {kind: 0 for kind in arm_rows}
        clips_block: list[dict[str, Any]] = []
        for clip in test_clips:
            features = features_by_clip[clip.clip_id]
            gt = list(gt_by_clip[clip.clip_id])
            estimator = [int(v) for v in estimator_by_clip[clip.clip_id].tolist()]
            candidate_r, _ = causal_gate_trace(gate, features, theta, state_factory)
            arm_r = {
                ARM_CANDIDATE: candidate_r,
                ARM_RATE_MATCHED_RANDOM: rate_matched_random_fires(
                    candidate_r, clip.n_frames, seed=seed, clip_id=clip.clip_id
                ),
                ARM_ALWAYS_ON: always_on_fires(clip.n_frames),
                ARM_NEVER_UPDATE: never_update_reestimates(clip.n_frames),
            }
            for kind, r in arm_r.items():
                arm_rows[kind].append((clip.clip_id, gt, estimator, list(r)))
                reestimations[kind] += len(r)
            clips_block.append(
                {
                    "clip_id": clip.clip_id,
                    "reestimate_frames": {
                        ARM_CANDIDATE: list(arm_r[ARM_CANDIDATE]),
                        ARM_RATE_MATCHED_RANDOM: list(arm_r[ARM_RATE_MATCHED_RANDOM]),
                    },
                }
            )
        arm_scores = {kind: score_rows(rows) for kind, rows in arm_rows.items()}
        per_budget[budget_id] = {
            "theta": theta,
            "rate": rate,
            "clips": clips_block,
            "arm_scores": arm_scores,
            "reestimations": reestimations,
        }

    operating_budget_id = min(per_budget, key=lambda bid: abs(per_budget[bid]["rate"] - operating_density))
    operating = per_budget[operating_budget_id]
    per_seed_block = {
        "seed": seed,
        "operating_budget_id": operating_budget_id,
        "clips": operating["clips"],
        "arm_scores": operating["arm_scores"],
    }

    operating_theta = operating["theta"]
    base_rate = operating["reestimations"][ARM_CANDIDATE] / max(1, total_frames)
    noise_reestimates, _ = causal_gate_trace(gate, noise_features, operating_theta, state_factory)
    noise_rate = len(noise_reestimates) / noise_features.shape[0]
    noisy_tv = {
        "reestimate_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    return BudgetSeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
    )


def _micro_count_score(
    rows: list[tuple[str, list[int], list[int], list[int]]],
) -> dict[str, Any]:
    return score_arm([(gt, estimator, r) for _, gt, estimator, r in rows], COLD_START).payload()


def _run_seed_real(
    seed: int,
    train_clips: tuple[Clip, ...],
    val_clips: tuple[Clip, ...],
    test_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    estimator_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    noise_features: np.ndarray,
    config: RealCountBedConfig,
    operating_density: float,
    *,
    train_gate_provider: Callable[..., tuple[Any, int]] = _train_count_gate,
) -> BudgetSeedRun:

    return run_count_seed(
        seed=seed,
        val_clips=val_clips,
        test_clips=test_clips,
        features_by_clip=features_by_clip,
        estimator_by_clip=estimator_by_clip,
        gt_by_clip=gt_by_clip,
        noise_features=noise_features,
        target_rates=config.target_rates,
        operating_density=operating_density,
        train_gate=lambda: train_gate_provider(seed, train_clips, features_by_clip, gt_by_clip, config),
        state_factory=CountOnlineState.initial,
        score_rows=_micro_count_score,
    )


def _flop_model(kind: str, total_frames: int, train_frames: int, config: RealCountBedConfig) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME_COUNT,
        gate_infer_per_frame=FLOPS_PER_INFERENCE,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        candidate_train_flops=lambda: training_flops(train_frames, config.epochs),
    )


def build_real_count_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_PREREG_PATH,
) -> ArtifactResult:

    config = config or RealCountBedConfig()
    featurizer = FrozenCountFeaturizer()
    estimator = FrozenCountEstimator()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    corpus = prepare_count_variant_corpus(
        adapter=adapter,
        foa_root=foa_root,
        metadata_root=metadata_root,
        featurizer=featurizer,
        estimator=estimator,
        config=config,
        split_provider=lambda current_adapter: native_fold_split(
            current_adapter,
            config.n_val_rooms,
            refusal=CountProducerRefusal,
        ),
    )

    def featurizer_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
        }

    def gate_payload(context: CountVariantContext) -> dict[str, Any]:
        return {
            "params": context.seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        }

    def estimator_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        }

    spec = CountVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        producer_schema=COUNT_PRODUCER_SCHEMA,
        refusal=CountProducerRefusal,
        no_changes_message="the real test split carries no count changes to track",
        score_field="mae",
        build_prereg=lambda current: build_count_prereg(
            timestamp=timestamp,
            operating_reestimate_fraction=current.operating_rate,
            n_test_clips=current.n_test_clips,
            n_test_changes=current.n_test_changes,
            n_test_frames=current.n_test_frames,
            train_change_density=current.train_density,
            coast_from_zero_mae=current.test_coast_from_zero,
        ),
        noise_features=_real_noisy_tv_features,
        run_seed=lambda seed, current, noise: _run_seed_real(
            seed,
            current.train_clips,
            current.val_clips,
            current.test_clips,
            current.features_by_clip,
            current.estimator_by_clip,
            current.gt_by_clip,
            noise,
            config,
            current.train_density,
        ),
        flop_model=lambda kind, total_frames, train_frames: _flop_model(
            kind, total_frames, train_frames, config
        ),
        featurizer_payload=featurizer_payload,
        gate_payload=gate_payload,
        estimator_payload=estimator_payload,
        receipt_detail=lambda _context: {
            "question": (
                "concurrent-source counting (distinct from the seven sealed onset-localization nulls)"
            ),
            "note": (
                "one real run is a mechanics demonstration; scientific confirmation needs the independent "
                "verifier plus at least three bias-independent reproductions and cannot be self-certified"
            ),
        },
        artifact_extra=lambda _context: {
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
                "break_even_frames_anchor": (FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate),
            },
        },
        prereg_extra=lambda _context: {},
        final_extra=lambda _context: {},
    )
    return build_count_variant_artifact(
        config=config,
        corpus=corpus,
        featurizer=featurizer,
        estimator=estimator,
        prereg_path=prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )


DEFAULT_COUNT_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_BED.json")
