from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.evidence import write_canonical_json
from mop.science import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    ArtifactResult,
    artifact_envelope,
    demonstration_receipt,
    finalize_artifact,
    safety_flags,
)
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_exceeded

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import map_clip_audio
from .controls import at_chance
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_referee import COLD_START
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY


def _no_analysis(_seed_runs: list[Any]) -> None:
    return None


def _empty_analysis_payload(_analysis: Any) -> dict[str, Any]:
    return {}


def _analysis_passes(_analysis: Any) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class CountVariantCorpus:
    adapter: Any
    foa_root: Path
    metadata_root: Path
    train_clips: tuple[Any, ...]
    val_clips: tuple[Any, ...]
    test_clips: tuple[Any, ...]
    test_count_clips: tuple[Any, ...]
    split_detail: dict[str, Any]
    features_by_clip: dict[str, np.ndarray]
    estimator_by_clip: dict[str, np.ndarray]
    gt_by_clip: dict[str, tuple[int, ...]]
    train_density: float
    n_test_clips: int
    n_test_frames: int
    n_test_changes: int
    test_coast_from_zero: float
    operating_rate: float


@dataclass(frozen=True, slots=True)
class CountVariantSpec:
    artifact_schema: str
    producer_schema: str
    refusal: type[ValueError]
    no_changes_message: str
    score_field: str
    build_prereg: Callable[[CountVariantCorpus], dict[str, Any]]
    noise_features: Callable[[int, int, Any, float, float], np.ndarray]
    run_seed: Callable[[int, CountVariantCorpus, np.ndarray], Any]
    flop_model: Callable[[str, int, int], Any]
    featurizer_payload: Callable[[CountVariantContext], dict[str, Any]]
    gate_payload: Callable[[CountVariantContext], dict[str, Any]]
    estimator_payload: Callable[[CountVariantContext], dict[str, Any]]
    receipt_detail: Callable[[CountVariantContext], dict[str, Any]]
    artifact_extra: Callable[[CountVariantContext], dict[str, Any]]
    prereg_extra: Callable[[CountVariantContext], dict[str, Any]]
    final_extra: Callable[[CountVariantContext], dict[str, Any]]
    analyze_seed_runs: Callable[[list[Any]], Any] = _no_analysis
    stats_options: Callable[[Any], dict[str, Any]] = _empty_analysis_payload
    core_evidence_extra: Callable[[Any], dict[str, Any]] = _empty_analysis_payload
    analysis_survives: Callable[[Any], bool] = _analysis_passes


@dataclass(frozen=True, slots=True)
class CountVariantContext:
    seed_runs: list[Any]
    report: Any
    sign_flip: Any
    mean_delta_exceeds_sesoi: bool
    analysis: Any


def prepare_count_variant_corpus(
    *,
    adapter: Any,
    foa_root: str | Path,
    metadata_root: str | Path,
    featurizer: Any,
    estimator: Any,
    config: Any,
    split_provider: Callable[[Any], Any],
) -> CountVariantCorpus:

    count_clips = build_count_clips(adapter, metadata_root)
    gt_by_clip = {clip_id: clip.count_track for clip_id, clip in count_clips.items()}
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    estimator_by_clip = map_clip_audio(adapter, estimator.estimate_track)
    split = split_provider(adapter)
    if hasattr(split, "train"):
        train_clips, val_clips, test_clips = split.train, split.val, split.test
        split_detail = dict(split.detail)
    else:
        train_clips, val_clips, test_clips, detail = split
        split_detail = dict(detail)
    train_count_clips = [count_clips[clip.clip_id] for clip in train_clips]
    test_count_clips = [count_clips[clip.clip_id] for clip in test_clips]
    train_density = change_density(train_count_clips)
    return CountVariantCorpus(
        adapter=adapter,
        foa_root=Path(foa_root),
        metadata_root=Path(metadata_root),
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        test_count_clips=tuple(test_count_clips),
        split_detail=split_detail,
        features_by_clip=features_by_clip,
        estimator_by_clip=estimator_by_clip,
        gt_by_clip=gt_by_clip,
        train_density=train_density,
        n_test_clips=len(test_clips),
        n_test_frames=int(sum(clip.n_frames for clip in test_clips)),
        n_test_changes=int(sum(clip.n_changes for clip in test_count_clips)),
        test_coast_from_zero=coast_from_zero_mae(test_count_clips),
        operating_rate=min(config.target_rates, key=lambda rate: abs(rate - train_density)),
    )


def build_count_variant_artifact(
    *,
    config: Any,
    corpus: CountVariantCorpus,
    featurizer: Any,
    estimator: Any,
    prereg_path: str | Path,
    spec: CountVariantSpec,
    clock_ns: Callable[[], int],
) -> ArtifactResult:

    if corpus.n_test_changes == 0:
        raise spec.refusal(spec.no_changes_message)
    prereg = spec.build_prereg(corpus)
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_mae = float(prereg["sesoi"]["sesoi_mae"])

    pooled_test_features = np.concatenate(
        [corpus.features_by_clip[clip.clip_id] for clip in corpus.test_clips], axis=0
    )
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())
    started = clock_ns()
    seed_runs = [
        spec.run_seed(
            seed,
            corpus,
            spec.noise_features(seed, config.noisy_tv_frames, featurizer, target_mean, target_std),
        )
        for seed in config.seeds
    ]
    measured_wall_ns = max(1, clock_ns() - started)

    report = run_matched_budget(
        COUNT_BUDGET_POLICY,
        seed_runs,
        score_group="arm_scores",
        score_field=spec.score_field,
        action_group="reestimations",
        flop_model=lambda kind: spec.flop_model(kind, seed_runs[0].total_frames, seed_runs[0].train_frames),
        operating_budget_id=seed_runs[0].operating_budget_id,
        source_kind="real",
        ceiling=FLOP_CEILING,
    )
    per_seed = [run.per_seed_block for run in seed_runs]
    deltas = [
        block["arm_scores"][ARM_RATE_MATCHED_RANDOM][spec.score_field]
        - block["arm_scores"][ARM_CANDIDATE][spec.score_field]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    analysis = spec.analyze_seed_runs(seed_runs)
    mean_delta_exceeds_sesoi = sesoi_exceeded(sign_flip.mean_delta, sesoi_mae)
    mean_delta_candidate_minus_random = -float(sign_flip.mean_delta)
    stats = count_sign_flip_payload(
        sign_flip,
        deltas,
        sesoi=sesoi_mae,
        exceeds_sesoi=mean_delta_exceeds_sesoi,
        mean_candidate_minus_control=mean_delta_candidate_minus_random,
        prereg_digest=prereg["canonical_sha256"],
        **spec.stats_options(analysis),
    )
    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["reestimate_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls = noise_control_summary(
        COUNT_BUDGET_POLICY,
        seed_runs,
        at_chance=noisy_tv_at_chance,
        mean_noise_rate=mean_noise_rate,
        mean_base_rate=mean_base_rate,
        rate_key="mean_reestimate_rate_on_noise",
    )
    flags = safety_flags()
    corpus_tracks = {
        clip.clip_id: {
            "n_frames": clip.n_frames,
            "gt_count_track": list(corpus.gt_by_clip[clip.clip_id]),
            "estimator_track": [int(value) for value in corpus.estimator_by_clip[clip.clip_id].tolist()],
        }
        for clip in corpus.test_clips
    }
    dominates = report.candidate_strictly_dominates_rate_matched_random
    meets_bar = (
        dominates
        and sign_flip.one_sided_significant
        and mean_delta_exceeds_sesoi
        and spec.analysis_survives(analysis)
    )
    verdict = VERDICT_MECHANICS_OK if meets_bar else VERDICT_NULL
    truncations = [truncation.payload() for truncation in corpus.adapter.truncations()]
    context = CountVariantContext(
        seed_runs=seed_runs,
        report=report,
        sign_flip=sign_flip,
        mean_delta_exceeds_sesoi=mean_delta_exceeds_sesoi,
        analysis=analysis,
    )
    core_evidence = {
        "per_seed": per_seed,
        "stats": stats,
        **spec.core_evidence_extra(analysis),
        "controls": controls,
        "matched_budget": report.matched_budget.payload(),
        "flags": flags,
    }
    receipt = demonstration_receipt(
        mechanism_id=COUNT_BED_ID,
        controls_cleared=(
            ARM_RATE_MATCHED_RANDOM,
            ARM_ALWAYS_ON,
            ARM_NEVER_UPDATE,
            "noisy_tv",
        ),
        evidence=core_evidence,
        verdict=verdict,
        detail={
            "source_kind": "real",
            "forcing_null": STAGE3_FORCING_NULL,
            **spec.receipt_detail(context),
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
        },
    )
    dropped_onsets = sum(item["dropped_onsets_past_end"] for item in truncations)
    capped_clips = sum(1 for item in truncations if item["capped_by_max_frames"])
    real_corpus = {
        "producer_schema": spec.producer_schema,
        "foa_root": str(corpus.foa_root),
        "metadata_root": str(corpus.metadata_root),
        "n_clips": len(corpus.adapter.clips()),
        "split_rooms": corpus.split_detail,
        "n_train_clips": len(corpus.train_clips),
        "n_val_clips": len(corpus.val_clips),
        "n_train_frames": seed_runs[0].train_frames,
        "n_test_clips": corpus.n_test_clips,
        "n_test_frames": corpus.n_test_frames,
        "n_test_changes": corpus.n_test_changes,
        "train_change_density": round(float(corpus.train_density), 12),
        "test_coast_from_zero_mae": round(float(corpus.test_coast_from_zero), 12),
        "operating_reestimate_fraction": round(float(corpus.operating_rate), 12),
        "truncation": {
            "clips_capped_by_max_frames": capped_clips,
            "onsets_dropped_past_audio_end": dropped_onsets,
            "max_frames": config.max_frames,
            "per_clip": truncations,
        },
    }
    prereg_payload = {
        "path": str(prereg_written),
        "canonical_sha256": prereg["canonical_sha256"],
        "sesoi_mae": sesoi_mae,
        **spec.prereg_extra(context),
        "provisional": False,
        "written_before_test_scores": True,
    }
    body = artifact_envelope(
        schema=spec.artifact_schema,
        report=report,
        seeds=config.seeds,
        per_seed=per_seed,
        stats=stats,
        controls=controls,
        flags=flags,
        verdict=verdict,
        featurizer=spec.featurizer_payload(context),
        gate=spec.gate_payload(context),
        receipt_payload=receipt,
        extra={
            **spec.artifact_extra(context),
            "cold_start": COLD_START,
            "primary_control": ARM_RATE_MATCHED_RANDOM,
            "corpus_tracks": corpus_tracks,
            "estimator": spec.estimator_payload(context),
            "real_corpus": real_corpus,
            "prereg": prereg_payload,
        },
    )
    return finalize_artifact(
        body,
        prereg=prereg,
        verdict=verdict,
        detail={
            "dominates": dominates,
            "mean_delta_control_minus_candidate": float(sign_flip.mean_delta),
            "mean_delta_candidate_minus_control": mean_delta_candidate_minus_random,
            "one_sided_p": float(sign_flip.one_sided_p),
            "one_sided_significant": bool(sign_flip.one_sided_significant),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_mae": sesoi_mae,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(value) for value in deltas],
            **spec.final_extra(context),
        },
    )


__all__ = [
    "CountVariantContext",
    "CountVariantCorpus",
    "CountVariantSpec",
    "build_count_variant_artifact",
    "prepare_count_variant_corpus",
]
