
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.ladder.ladder_contracts import VERDICT_MECHANICS_OK, VERDICT_NULL
from mop.science import (
    ArtifactResult,
    artifact_envelope,
    demonstration_receipt,
    finalize_artifact,
    read_sealed_prereg_member,
    safety_flags,
)
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    arm_flop_model,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import exact_sign_flip, sign_flip_payload

from . import BED_ID, FLOP_CEILING, STAGE3_FORCING_NULL
from .artifact import PRIMARY_CONTROL
from .controls import at_chance
from .experiments import ONSET_BUDGET_POLICY
from .gate import FLOPS_PER_INFERENCE, OnlineState, training_flops
from .real_artifact import _real_noisy_tv_features, _run_seed_real
from .referee import summarize_fire_spread_blocks
from .schema import COLLAR_FRAMES

_DEFAULT_RECEIPT_NOTE = (
    "one real run of one frozen featurizer is a mechanics outcome; scientific confirmation "
    "needs the independent verifier plus at least three bias-independent reproductions and, "
    "for this three-featurizer family at n equals 5, cannot clear family-wise significance"
)


def _default_seed_run(
    seed: int,
    corpus: VariantCorpus,
    noise_features: np.ndarray,
    bed_config: Any,
) -> Any:
    return _run_seed_real(
        seed,
        corpus.split,
        corpus.features_by_clip,
        noise_features,
        bed_config,
        corpus.train_density,
    )


def _beats_random_extra(beats_random: bool) -> dict[str, Any]:
    return {"beats_rate_matched_random": beats_random}


@dataclass(frozen=True, slots=True)
class VariantCorpus:

    split: Any
    features_by_clip: dict[str, np.ndarray]
    train_density: float
    n_test_clips: int
    n_test_onsets: int
    n_test_frames: int


@dataclass(frozen=True, slots=True)
class FeaturizerVariantSpec:

    artifact_schema: str
    variant_id: str
    identity_key: str | None
    prereg_schema: str
    prereg_family_field: str
    prereg_member_field: str
    refusal: type[ValueError]
    flops_per_frame: int
    spread: Callable[[list[dict[str, Any]]], dict[str, Any]]
    featurizer_payload: Callable[[VariantContext], dict[str, Any]]
    extra_payload: Callable[[VariantContext], dict[str, Any]]
    final_extra: Callable[[VariantContext], dict[str, Any]]
    receipt_extra: dict[str, Any]
    run_seed: Callable[[int, VariantCorpus, np.ndarray, Any], Any] = _default_seed_run
    prereg_family_label: str = "featurizer"
    receipt_note: str = _DEFAULT_RECEIPT_NOTE
    prepare_prereg: Callable[[VariantCorpus], tuple[dict[str, Any], str | Path]] | None = None
    include_prereg_in_result: bool = False
    flop_model: Callable[[str, list[Any], Any], Any] | None = None
    include_spread_in_detail: bool = True
    gate_payload: Callable[[VariantContext], dict[str, Any]] | None = None
    stats_extra: Callable[[bool], dict[str, Any]] = _beats_random_extra
    include_beats_random_in_detail: bool = True


@dataclass(frozen=True, slots=True)
class VariantContext:

    prereg_digest: str
    sesoi_f1: float
    operating_rate: float
    seed_runs: list[Any]
    budget_points: list[Any]
    report: Any
    beats_random: bool
    spread: dict[str, Any]


def featurizer_spread_diagnostic(
    per_seed: list[dict[str, Any]],
    *,
    definition: str,
    anchor_key: str,
    source: str,
) -> dict[str, Any]:

    return {
        "definition": definition,
        "collar_frames": COLLAR_FRAMES,
        "candidate": summarize_fire_spread_blocks(per_seed, ARM_CANDIDATE),
        "rate_matched_random": summarize_fire_spread_blocks(per_seed, ARM_RATE_MATCHED_RANDOM),
        anchor_key: {
            "candidate_distinct_onset_tp": 204,
            "rate_matched_random_distinct_onset_tp": 237,
            "candidate_adjacency_fraction_approx": 0.42,
            "source": source,
        },
    }


def build_featurizer_variant_artifact(
    *,
    config: Any,
    bed_config: Any,
    corpus: VariantCorpus,
    featurizer: Any,
    prereg_path: str | Path,
    spec: FeaturizerVariantSpec,
    clock_ns: Callable[[], int],
) -> ArtifactResult:

    if spec.prepare_prereg is None:
        prereg = read_sealed_prereg_member(
            prereg_path,
            expected_schema=spec.prereg_schema,
            family_field=spec.prereg_family_field,
            member_field=spec.prereg_member_field,
            member_id=spec.variant_id,
            family_label=spec.prereg_family_label,
            refusal=spec.refusal,
        )
    else:
        prereg, prereg_path = spec.prepare_prereg(corpus)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])
    prereg_digest = str(prereg["canonical_sha256"])
    if corpus.n_test_onsets == 0:
        raise spec.refusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda rate: abs(rate - corpus.train_density))

    pooled_test_features = np.concatenate(
        [corpus.features_by_clip[clip.clip_id] for clip in corpus.split.test], axis=0
    )
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = clock_ns()
    seed_runs: list[Any] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_runs.append(spec.run_seed(seed, corpus, noise_features, bed_config))
    measured_wall_ns = max(1, clock_ns() - started)

    budget_points = build_budget_points(
        ONSET_BUDGET_POLICY,
        seed_runs,
        score_group="arm_scores",
        score_field="f1",
        action_group="firings",
        flop_model=lambda kind: (
            spec.flop_model(kind, seed_runs, bed_config)
            if spec.flop_model is not None
            else arm_flop_model(
                kind,
                seed_runs[0].total_frames,
                featurize_per_frame=spec.flops_per_frame,
                gate_infer_per_frame=FLOPS_PER_INFERENCE,
                downstream_flops_per_firing=bed_config.downstream_flops_per_firing,
                candidate_train_flops=lambda: training_flops(
                    seed_runs[0].train_frames, bed_config.epochs
                ),
            )
        ),
    )
    nominal_wall_ns = max(
        1, max(point.candidate.max_lifecycle_flops() for point in budget_points)
    )
    report = run_matched_budget(
        budget_points,
        wall_ns=nominal_wall_ns,
        operating_budget_id=seed_runs[0].operating_budget_id,
        source_kind="real",
        ceiling=FLOP_CEILING,
    )

    per_seed = [run.per_seed_block for run in seed_runs]
    deltas = [
        block["arm_scores"][ARM_CANDIDATE]["f1"]
        - block["arm_scores"][PRIMARY_CONTROL]["f1"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    mean_delta_exceeds_sesoi = bool(sign_flip.mean_delta >= sesoi_f1)
    beats_random = bool(sign_flip.one_sided_significant and mean_delta_exceeds_sesoi)
    stats = sign_flip_payload(
        sign_flip,
        deltas,
        sesoi_key="sesoi_f1",
        sesoi=sesoi_f1,
        exceeds_sesoi=mean_delta_exceeds_sesoi,
        provisional=False,
        prereg_digest=prereg_digest,
        extra=spec.stats_extra(beats_random),
    )

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["firing_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls = noise_control_summary(
        ONSET_BUDGET_POLICY,
        seed_runs,
        at_chance=noisy_tv_at_chance,
        mean_noise_rate=mean_noise_rate,
        mean_base_rate=mean_base_rate,
        rate_key="mean_firing_rate_on_noise",
    )
    flags = safety_flags()
    spread = spec.spread(per_seed)
    dominates = report.candidate_strictly_dominates_rate_matched_random
    meets_bar = dominates and sign_flip.one_sided_significant and mean_delta_exceeds_sesoi
    verdict = VERDICT_MECHANICS_OK if meets_bar else VERDICT_NULL

    context = VariantContext(
        prereg_digest=prereg_digest,
        sesoi_f1=sesoi_f1,
        operating_rate=operating_rate,
        seed_runs=seed_runs,
        budget_points=budget_points,
        report=report,
        beats_random=beats_random,
        spread=spread,
    )
    core_evidence = {
        "per_seed": per_seed,
        "stats": stats,
        "controls": controls,
        "matched_budget": report.matched_budget.payload(),
        "flags": flags,
    }
    receipt_detail = {
        "source_kind": "real",
        **spec.receipt_extra,
        "forcing_null": STAGE3_FORCING_NULL,
        "candidate_strictly_dominates_rate_matched_random": dominates,
        "one_sided_p": float(sign_flip.one_sided_p),
        "note": spec.receipt_note,
    }
    if spec.identity_key is not None:
        receipt_detail[spec.identity_key] = spec.variant_id
    receipt = demonstration_receipt(
        mechanism_id=BED_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv"),
        evidence=core_evidence,
        verdict=verdict,
        detail=receipt_detail,
    )
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
        gate=(
            spec.gate_payload(context)
            if spec.gate_payload is not None
            else {
                "params": seed_runs[0].gate_params,
                "param_ceiling": 4096,
                "state_bytes": OnlineState.state_bytes(),
                "flops_per_inference": FLOPS_PER_INFERENCE,
            }
        ),
        receipt_payload=receipt,
        extra=spec.extra_payload(context),
    )
    detail = {
        "dominates": dominates,
        "mean_delta": float(sign_flip.mean_delta),
        "one_sided_p": float(sign_flip.one_sided_p),
        "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
        "sesoi_f1": sesoi_f1,
        "noisy_tv_at_chance": noisy_tv_at_chance,
        "measured_wall_ns": measured_wall_ns,
        "per_seed_deltas": [float(value) for value in deltas],
        **spec.final_extra(context),
    }
    if spec.include_beats_random_in_detail:
        detail["beats_random"] = beats_random
    if spec.include_spread_in_detail:
        detail["spread"] = spread
    kwargs = {"prereg": prereg} if spec.include_prereg_in_result else {}
    return finalize_artifact(body, verdict=verdict, detail=detail, **kwargs)


__all__ = [
    "FeaturizerVariantSpec",
    "VariantContext",
    "VariantCorpus",
    "build_featurizer_variant_artifact",
    "featurizer_spread_diagnostic",
]
