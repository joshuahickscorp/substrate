"""Clip-macro scoring reproduction producer for the STARSS23 concurrent-count bed.

The scoring unit, clustered readout, preregistration, and evidence vocabulary remain local; the shared
count-variant authority owns only the held-fixed execution and sealed-artifact lifecycle."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    BudgetSeedRun,
)

from .adapter import RealStarssAdapter, native_fold_split
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import (
    FLOPS_PER_INFERENCE,
    CountOnlineState,
)
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    CountProducerRefusal,
    RealCountBedConfig,
    _flop_model,
    _real_noisy_tv_features,
    _train_count_gate,
    run_count_seed,
)
from .count_referee import COLD_START
from .count_repro_scoring_unit_prereg import (
    DEFAULT_COUNT_REPRO_SCORING_UNIT_PREREG_PATH,
    ClipLabelFact,
    build_count_repro_scoring_unit_prereg,
)
from .count_repro_scoring_unit_referee import (
    SCORING_UNIT,
    exact_sign_flip_over_clips,
    macro_score_arm,
)
from .count_variant_producer import (
    CountVariantContext,
    CountVariantSpec,
    build_count_variant_artifact,
    prepare_count_variant_corpus,
)
from .experiments import COUNT_BED_ID
from .schema import Clip

COUNT_REPRO_SCORING_UNIT_PRODUCER_SCHEMA = "mop-starss23-count-repro-scoring-unit-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-repro-scoring-unit-bed/v1"
STAGE = 3
REPRO_AXIS = "scoring_unit"
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The disjoint seed family for this reproduction: it must not inherit the sealed run's seed luck.
DEFAULT_SCORING_UNIT_SEEDS: tuple[int, ...] = (30, 31, 32, 33, 34)

class CountReproScoringUnitProducerRefusal(ValueError):
    """Raised when the clip-macro reproduction cannot assemble a well-formed sealed artifact."""


def _macro_count_score(
    rows: list[tuple[str, list[int], list[int], list[int]]],
) -> dict[str, Any]:
    return macro_score_arm(rows, COLD_START).payload()


def default_scoring_unit_config() -> RealCountBedConfig:
    """The full-scale reproduction config: the disjoint seed family, everything else at sealed defaults."""

    return RealCountBedConfig(seeds=DEFAULT_SCORING_UNIT_SEEDS)


def _run_seed_macro(
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
) -> BudgetSeedRun:
    """Bind the clip-macro referee to the held-fixed counting seed lifecycle."""

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
        score_rows=_macro_count_score,
    )


def _clip_cluster_readout(seed_runs: list[BudgetSeedRun]) -> tuple[dict[str, Any], bool]:
    """Form per-clip paired deltas at the operating point (per-clip MAE averaged over seeds) and permute.

    d_c = mean_seed macro MAE_clip(rate_matched_random, c) minus mean_seed macro MAE_clip(candidate, c). A
    positive d_c means the candidate is strictly lower on clip c on the seed average. The exact sign-flip
    over clips assigns one sign per clip. Direction agreement requires the mean per-clip delta positive.
    """

    operating_id = seed_runs[0].operating_budget_id
    n_seeds = len(seed_runs)
    # Collect the per-clip MAE for candidate and control at the operating point, per seed.
    clip_ids = [
        entry["clip_id"]
        for entry in seed_runs[0].per_budget[operating_id]["clips"]
    ]
    cand_sum: dict[str, float] = {cid: 0.0 for cid in clip_ids}
    rmr_sum: dict[str, float] = {cid: 0.0 for cid in clip_ids}
    for run in seed_runs:
        arm_scores = run.per_budget[operating_id]["arm_scores"]
        cand_per_clip = arm_scores[ARM_CANDIDATE]["per_clip"]
        rmr_per_clip = arm_scores[ARM_RATE_MATCHED_RANDOM]["per_clip"]
        for cid in clip_ids:
            cand_sum[cid] += float(cand_per_clip[cid]["mae"])
            rmr_sum[cid] += float(rmr_per_clip[cid]["mae"])
    deltas = [(rmr_sum[cid] / n_seeds) - (cand_sum[cid] / n_seeds) for cid in clip_ids]
    permutation = exact_sign_flip_over_clips(deltas)
    readout = {
        "operating_budget_id": operating_id,
        "per_clip_delta": {cid: round(float(d), 12) for cid, d in zip(clip_ids, deltas, strict=True)},
        "permutation": permutation.payload(),
    }
    return readout, bool(permutation.direction_agrees)


def build_real_count_repro_scoring_unit_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_REPRO_SCORING_UNIT_PREREG_PATH,
) -> ArtifactResult:
    """Run the clip-macro scoring declaration through the shared count lifecycle."""

    config = config or default_scoring_unit_config()
    featurizer = FrozenCountFeaturizer()
    estimator = FrozenCountEstimator()
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

    def survives(context: CountVariantContext) -> bool:
        return bool(
            context.report.candidate_strictly_dominates_rate_matched_random
            and context.sign_flip.one_sided_significant
            and context.mean_delta_exceeds_sesoi
            and context.analysis[1]
        )

    def artifact_extra(context: CountVariantContext) -> dict[str, Any]:
        clip_cluster, _direction_agrees = context.analysis
        return {
            "reproduction_axis": REPRO_AXIS,
            "reproduces": COUNT_BED_ID,
            "scoring_unit": SCORING_UNIT,
            "clip_cluster": clip_cluster,
            "survive": survives(context),
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate
                ),
            },
        }

    def final_extra(context: CountVariantContext) -> dict[str, Any]:
        clip_cluster, direction_agrees = context.analysis
        return {
            "survive": survives(context),
            "clip_cluster_direction_agrees": direction_agrees,
            "clip_cluster_one_sided_p": float(
                clip_cluster["permutation"]["one_sided_p"]
            ),
        }

    spec = CountVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        producer_schema=COUNT_REPRO_SCORING_UNIT_PRODUCER_SCHEMA,
        refusal=CountReproScoringUnitProducerRefusal,
        no_changes_message="the real test split carries no count changes to track",
        score_field="macro_mae",
        build_prereg=lambda current: build_count_repro_scoring_unit_prereg(
            timestamp=timestamp,
            operating_reestimate_fraction=current.operating_rate,
            test_clip_facts=tuple(
                ClipLabelFact(
                    clip_id=clip.clip_id,
                    n_frames=clip.n_frames,
                    n_changes=clip.n_changes,
                )
                for clip in current.test_count_clips
            ),
            train_change_density=current.train_density,
            coast_from_zero_mae=current.test_coast_from_zero,
        ),
        noise_features=_real_noisy_tv_features,
        run_seed=lambda seed, current, noise: _run_seed_macro(
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
        receipt_detail=lambda context: {
            "reproduction_axis": REPRO_AXIS,
            "question": (
                "does the sealed counting win survive with the clip as the experimental unit "
                "(clip-macro scoring) instead of the pooled frame micro-average"
            ),
            "clip_cluster_direction_agrees": context.analysis[1],
            "note": (
                "one real reproduction is a mechanics demonstration; scientific confirmation needs the "
                "independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified"
            ),
        },
        artifact_extra=artifact_extra,
        prereg_extra=lambda _context: {
            "sesoi_scale": "clip-macro count-MAE",
        },
        final_extra=final_extra,
        analyze_seed_runs=_clip_cluster_readout,
        stats_options=lambda analysis: {
            "metric": "coasted-count-MAE (clip-macro)",
            "delta_definition": (
                "delta_i = macro_MAE_rate_matched_random(i) - macro_MAE_candidate(i); positive = "
                "candidate lower clip-macro error"
            ),
            "extra": {
                "scoring_unit": SCORING_UNIT,
                "sesoi_scale": "clip-macro count-MAE",
                "clip_cluster_direction_agrees": analysis[1],
            },
        },
        core_evidence_extra=lambda analysis: {
            "clip_cluster": analysis[0],
        },
        analysis_survives=lambda analysis: analysis[1],
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


DEFAULT_COUNT_REPRO_SCORING_UNIT_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_scoring_unit.json")
