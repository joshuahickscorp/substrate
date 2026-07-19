"""Scoring-unit adversarial reproduction, component C: the clip-macro real-data producer.

Net-new and additive. It re-runs the sealed STARSS23 concurrent-source-counting bed end to end on the REAL
room-disjoint STARSS23 subset, changing exactly ONE axis relative to the sealed run: the scoring unit. The
sealed run pools a frame micro-average; this reproduction scores with the clip as the experimental unit
(the clip-macro mean of per-clip MAE) and additionally runs an exact clip-clustered permutation. Every
other axis is held byte-identical BY IMPORT from the sealed and net-new count modules, so the only thing
that can move the verdict is the pooling:

- the native fold-respecting room-disjoint split (fold-3 train, last two fold-3 rooms val, fold-4 test) is
  reused unchanged from ``adapter.native_fold_split``;
- the frozen zero-parameter featurizer and estimator, the trained gate, the value-of-computation targets,
  the causal re-estimation passes, the three controls plus the real noisy-TV channel, the matched-budget
  FLOP model, and the exact five-seed sign-flip are all reused unchanged;
- only the seed family is disjoint from the sealed run, ``(30, 31, 32, 33, 34)``, so this reproduction does
  not inherit the sealed run's seed luck (the sealed run's effect concentrated on one seed).

The SESOI is preregistered on the clip-macro scale before any test score is read (see
``count_repro_scoring_unit_prereg``); this producer writes the sealed prereg first and records its digest.
The verdict is a mechanics demonstration only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single run can never be scientifically
confirmed. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from mop.ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
)
from mop.science import (
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
    BudgetSeedRun,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_check
from mop.substrate.events import write_canonical_json

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split
from .controls import (
    at_chance,
)
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import (
    FLOPS_PER_INFERENCE,
    CountOnlineState,
)
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    CountProducerRefusal,
    RealCountBedConfig,
    _causal_reestimates,
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
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY
from .schema import Clip

COUNT_REPRO_SCORING_UNIT_PRODUCER_SCHEMA = "mop-starss23-count-repro-scoring-unit-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-repro-scoring-unit-bed/v1"
STAGE = 3
REPRO_AXIS = "scoring_unit"
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The disjoint seed family for this reproduction: it must not inherit the sealed run's seed luck.
DEFAULT_SCORING_UNIT_SEEDS: tuple[int, ...] = (30, 31, 32, 33, 34)

_ARM_KINDS = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE)


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
        causal_reestimates=_causal_reestimates,
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
    """Run the clip-macro reproduction on the real STARSS23 subset and assemble the sealed artifact."""

    config = config or default_scoring_unit_config()
    featurizer = FrozenCountFeaturizer()
    estimator = FrozenCountEstimator()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    count_clips = build_count_clips(adapter, metadata_root)
    gt_by_clip = {cid: cc.count_track for cid, cc in count_clips.items()}

    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    estimator_by_clip = map_clip_audio(adapter, estimator.estimate_track)
    split = native_fold_split(adapter, config.n_val_rooms, refusal=CountProducerRefusal)
    train_clips, val_clips, test_clips = split.train, split.val, split.test
    split_detail = dict(split.detail)

    # Label-only structural facts for the SESOI cost-benefit and the operating-point rule.
    train_count_clips = [count_clips[c.clip_id] for c in train_clips]
    test_count_clips = [count_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_count_clips)
    n_test_clips = len(test_clips)
    n_test_frames = int(sum(clip.n_frames for clip in test_clips))
    n_test_changes = int(sum(cc.n_changes for cc in test_count_clips))
    test_coast_from_zero = coast_from_zero_mae(test_count_clips)
    if n_test_changes == 0:
        raise CountReproScoringUnitProducerRefusal("the real test split carries no count changes to track")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))
    test_clip_facts = tuple(
        ClipLabelFact(clip_id=cc.clip_id, n_frames=cc.n_frames, n_changes=cc.n_changes)
        for cc in test_count_clips
    )

    # 1. Preregister the clip-macro SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_count_repro_scoring_unit_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        test_clip_facts=test_clip_facts,
        train_change_density=train_density,
        coast_from_zero_mae=test_coast_from_zero,
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_macro = float(prereg["sesoi"]["sesoi_mae"])

    # 2. Now run the paired seeds and score the test split with the clip as the unit.
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in test_clips], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[BudgetSeedRun] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_runs.append(
            _run_seed_macro(
                seed,
                train_clips,
                val_clips,
                test_clips,
                features_by_clip,
                estimator_by_clip,
                gt_by_clip,
                noise_features,
                config,
                train_density,
            )
        )
    measured_wall_ns = max(1, time.perf_counter_ns() - started)

    budget_points = build_budget_points(
        COUNT_BUDGET_POLICY, seed_runs, score_group="arm_scores", score_field="macro_mae",
        action_group="reestimations",
        flop_model=lambda kind: _flop_model(
            kind, seed_runs[0].total_frames, seed_runs[0].train_frames, config
        ),
    )
    nominal_wall_ns = max(1, max(point.candidate.max_lifecycle_flops() for point in budget_points))
    report = run_matched_budget(
        budget_points,
        wall_ns=nominal_wall_ns,
        operating_budget_id=seed_runs[0].operating_budget_id,
        source_kind="real",
        ceiling=FLOP_CEILING,
    )

    per_seed = [run.per_seed_block for run in seed_runs]
    # Primary statistic: five-seed exact sign-flip on the clip-macro deltas (control minus candidate).
    deltas = [
        block["arm_scores"][PRIMARY_CONTROL]["macro_mae"] - block["arm_scores"][ARM_CANDIDATE]["macro_mae"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1=sesoi_macro, provisional=False)
    mean_delta_exceeds_sesoi = bool(sesoi.exceeds_sesoi)
    mean_delta_candidate_minus_random = -float(sign_flip.mean_delta)

    # Corroborating clip-clustered readout at the operating point.
    clip_cluster, clip_cluster_direction_agrees = _clip_cluster_readout(seed_runs)

    stats_block = count_sign_flip_payload(
        sign_flip, deltas, sesoi=sesoi_macro, exceeds_sesoi=mean_delta_exceeds_sesoi,
        mean_candidate_minus_control=mean_delta_candidate_minus_random,
        prereg_digest=prereg["canonical_sha256"], metric="coasted-count-MAE (clip-macro)",
        delta_definition=(
            "delta_i = macro_MAE_rate_matched_random(i) - macro_MAE_candidate(i); positive = candidate lower "
            "clip-macro error"
        ),
        extra={"scoring_unit": SCORING_UNIT, "sesoi_scale": "clip-macro count-MAE",
               "clip_cluster_direction_agrees": clip_cluster_direction_agrees},
    )

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["reestimate_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls_block = noise_control_summary(
        COUNT_BUDGET_POLICY, seed_runs, at_chance=noisy_tv_at_chance, mean_noise_rate=mean_noise_rate,
        mean_base_rate=mean_base_rate, rate_key="mean_reestimate_rate_on_noise",
    )
    flags_block = safety_flags()

    corpus_tracks = {
        clip.clip_id: {
            "n_frames": clip.n_frames,
            "gt_count_track": list(gt_by_clip[clip.clip_id]),
            "estimator_track": [int(v) for v in estimator_by_clip[clip.clip_id].tolist()],
        }
        for clip in test_clips
    }

    dominates = report.candidate_strictly_dominates_rate_matched_random
    survive = bool(
        dominates
        and sign_flip.one_sided_significant
        and mean_delta_exceeds_sesoi
        and clip_cluster_direction_agrees
    )
    verdict = VERDICT_MECHANICS_OK if survive else VERDICT_NULL

    core_evidence = {
        "per_seed": per_seed,
        "stats": stats_block,
        "clip_cluster": clip_cluster,
        "controls": controls_block,
        "matched_budget": report.matched_budget.payload(),
        "flags": flags_block,
    }
    receipt = demonstration_receipt(
        mechanism_id=COUNT_BED_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"),
        evidence=core_evidence,
        verdict=verdict,
        detail={
            "source_kind": "real",
            "forcing_null": STAGE3_FORCING_NULL,
            "reproduction_axis": REPRO_AXIS,
            "question": (
                "does the sealed counting win survive with the clip as the experimental unit (clip-macro "
                "scoring) instead of the pooled frame micro-average"
            ),
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "clip_cluster_direction_agrees": clip_cluster_direction_agrees,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one real reproduction is a mechanics demonstration; scientific confirmation needs the "
                "independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified"
            ),
        },
    )

    truncations = [t.payload() for t in adapter.truncations()]
    dropped_onsets = sum(t["dropped_onsets_past_end"] for t in truncations)
    capped_clips = sum(1 for t in truncations if t["capped_by_max_frames"])

    body = artifact_envelope(
        schema=ARTIFACT_SCHEMA, report=report, seeds=config.seeds, per_seed=per_seed,
        stats=stats_block, controls=controls_block, flags=flags_block, verdict=verdict,
        featurizer={
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
        }, gate={
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        }, receipt_payload=receipt,
        extra={
        "reproduction_axis": REPRO_AXIS,
        "reproduces": COUNT_BED_ID,
        "cold_start": COLD_START,
        "scoring_unit": SCORING_UNIT,
        "primary_control": PRIMARY_CONTROL,
        "corpus_tracks": corpus_tracks,
        "clip_cluster": clip_cluster,
        "survive": survive,
        "estimator": {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate,
        },
        "real_corpus": {
            "producer_schema": COUNT_REPRO_SCORING_UNIT_PRODUCER_SCHEMA,
            "foa_root": str(Path(foa_root)),
            "metadata_root": str(Path(metadata_root)),
            "n_clips": len(adapter.clips()),
            "split_rooms": split_detail,
            "n_train_clips": len(train_clips),
            "n_val_clips": len(val_clips),
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_frames": n_test_frames,
            "n_test_changes": n_test_changes,
            "train_change_density": round(float(train_density), 12),
            "test_coast_from_zero_mae": round(float(test_coast_from_zero), 12),
            "operating_reestimate_fraction": round(float(operating_rate), 12),
            "truncation": {
                "clips_capped_by_max_frames": capped_clips,
                "onsets_dropped_past_audio_end": dropped_onsets,
                "max_frames": config.max_frames,
                "per_clip": truncations,
            },
        },
        "prereg": {
            "path": str(prereg_written),
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_mae": sesoi_macro,
            "sesoi_scale": "clip-macro count-MAE",
            "provisional": False,
            "written_before_test_scores": True,
        },
        },
    )
    return finalize_artifact(
        body,
        prereg=prereg,
        verdict=verdict,
        detail={
            "survive": survive,
            "dominates": dominates,
            "mean_delta_control_minus_candidate": float(sign_flip.mean_delta),
            "mean_delta_candidate_minus_control": mean_delta_candidate_minus_random,
            "one_sided_p": float(sign_flip.one_sided_p),
            "one_sided_significant": bool(sign_flip.one_sided_significant),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_mae": sesoi_macro,
            "clip_cluster_direction_agrees": clip_cluster_direction_agrees,
            "clip_cluster_one_sided_p": float(clip_cluster["permutation"]["one_sided_p"]),
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(v) for v in deltas],
        },
    )


DEFAULT_COUNT_REPRO_SCORING_UNIT_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_scoring_unit.json")
