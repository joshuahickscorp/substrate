
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_RATE_MATCHED_RANDOM,
    BudgetSeedRun,
    FlopModel,
    arm_flop_model,
)

from .adapter import RealStarssAdapter, domain_seed, marginal_matched_noise, native_fold_split
from .count_gate import (
    COUNT_VOC_WINDOW,
    FLOPS_PER_INFERENCE,
    CountOnlineState,
)
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    CountProducerRefusal,
    _micro_count_score,
    _train_count_gate,
    run_count_seed,
)
from .count_repro_featurizer_estimator_estimator import (
    COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
    FLOPS_PER_REESTIMATE,
    ReproCountEstimator,
)
from .count_repro_featurizer_estimator_featurizer import (
    COUNT_REPRO_FE_FEATURIZER_SCHEMA,
    D_CFEAT,
    FLOPS_PER_FRAME_COUNT,
    ReproCountFeaturizer,
)
from .count_repro_featurizer_estimator_prereg import (
    DEFAULT_REPRO_PREREG_PATH,
    REPRO_AXIS,
    REPRO_SEEDS,
    build_repro_prereg,
)
from .count_variant_producer import (
    CountVariantContext,
    CountVariantSpec,
    build_count_variant_artifact,
    prepare_count_variant_corpus,
)
from .gate import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PONDER_LAMBDA, training_flops
from .schema import Clip

COUNT_REPRO_FE_PRODUCER_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-featurizer-estimator/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

DEFAULT_N_VAL_ROOMS = 2


class CountReproProducerRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReproCountBedConfig:

    seeds: tuple[int, ...] = REPRO_SEEDS
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
            raise CountReproProducerRefusal("the reproduction needs at least two paired seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise CountReproProducerRefusal("paired seeds must be unique")
        if not self.target_rates:
            raise CountReproProducerRefusal("at least one re-estimation budget target rate is required")


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: ReproCountFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:

    noise_seed = domain_seed(
        seed,
        "mop.beds.starss23.count_repro_featurizer_estimator.noisy_tv",
        b"mop-starss23-count-repro-featurizer-estimator-noisy-tv-v1",
    )
    return marginal_matched_noise(noise_seed, n_frames, featurizer, target_mean, target_std)


def _run_seed_real(
    seed: int,
    train_clips: tuple[Clip, ...],
    val_clips: tuple[Clip, ...],
    test_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    estimator_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    noise_features: np.ndarray,
    config: ReproCountBedConfig,
    operating_density: float,
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
        train_gate=lambda: _train_count_gate(
            seed, train_clips, features_by_clip, gt_by_clip, config
        ),
        state_factory=CountOnlineState.initial,
        score_rows=_micro_count_score,
    )


def _flop_model(kind: str, total_frames: int, train_frames: int, config: ReproCountBedConfig) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME_COUNT,
        gate_infer_per_frame=FLOPS_PER_INFERENCE,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        candidate_train_flops=lambda: training_flops(train_frames, config.epochs),
    )


def build_repro_count_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: ReproCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_REPRO_PREREG_PATH,
) -> ArtifactResult:

    config = config or ReproCountBedConfig()
    featurizer = ReproCountFeaturizer()
    estimator = ReproCountEstimator()
    adapter = RealStarssAdapter(
        foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames
    )
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
            "schema": COUNT_REPRO_FE_FEATURIZER_SCHEMA,
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
            "note": "re-authored gammatone ERB filterbank plus causal relative spectral flux",
        }

    def gate_payload(context: CountVariantContext) -> dict[str, Any]:
        return {
            "params": context.seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
            "note": "held fixed, identical to the sealed count gate",
        }

    def estimator_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "schema": COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
            "note": (
                "re-authored cumulative-energy (proportion-of-variance) count estimator, BETA=0.90"
            ),
        }

    spec = CountVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        producer_schema=COUNT_REPRO_FE_PRODUCER_SCHEMA,
        refusal=CountReproProducerRefusal,
        no_changes_message="the real test split carries no count changes to track",
        score_field="mae",
        build_prereg=lambda current: build_repro_prereg(
            timestamp=timestamp,
            operating_reestimate_fraction=current.operating_rate,
            n_test_clips=current.n_test_clips,
            n_test_changes=current.n_test_changes,
            n_test_frames=current.n_test_frames,
            train_change_density=current.train_density,
            coast_from_zero_mae=current.test_coast_from_zero,
            c_reest_flops=config.downstream_flops_per_reestimate,
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
            "reproduction_axis": REPRO_AXIS,
            "question": (
                "concurrent-source counting, featurizer_estimator bias-independent reproduction"
            ),
            "note": (
                "one real reproduction is a mechanics demonstration; scientific confirmation needs the "
                "independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified"
            ),
        },
        artifact_extra=lambda _context: {
            "reproduction_axis": REPRO_AXIS,
            "reproduces": "proof/STARSS23_COUNTING_BED.json",
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


DEFAULT_REPRO_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_featurizer_estimator.json")
