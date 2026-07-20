from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from mop.config import REPO_ROOT
from mop.evidence import canonical_sha256, write_canonical_json
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    BudgetSeedRun,
    FlopModel,
    arm_flop_model,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.gating import assemble_causal_inputs, causal_gate_trace
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_exceeded

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, domain_seed, map_clip_audio, marginal_matched_noise, native_fold_split
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
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY
from .schema import Clip

COUNT_PRODUCER_SCHEMA = "mop-starss23-count-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed/v1"
STAGE = 3

DEFAULT_STARSS_ROOT = Path(os.environ.get("MOP_STARSS23_ROOT", REPO_ROOT / "data" / "starss23"))
DEFAULT_FOA_ROOT = DEFAULT_STARSS_ROOT / "foa_subset" / "foa_dev"
DEFAULT_METADATA_ROOT = DEFAULT_STARSS_ROOT / "metadata_dev_extracted" / "metadata_dev"

DEFAULT_N_VAL_ROOMS = 2

FULL_SCALE_TRAIN_FRAMES = 54_000
FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_C_TRAIN = training_flops(FULL_SCALE_TRAIN_FRAMES, DEFAULT_EPOCHS)  # ~8.27e9
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME_COUNT * FULL_SCALE_TEST_FRAMES
MATCHED_BUDGET_WALL_NOTE = (
    "wall_ns is a deterministic nominal at a 1 GFLOP/s reference so the artifact is byte-reproducible; "
    "the measured wall is unsealed run provenance, and the authoritative sealed compute axes are the "
    "parameter count and the FLOP ledger"
)


class CountProducerRefusal(ValueError):
    pass


class CountArtifact(NamedTuple):
    artifact: dict[str, Any]
    prereg: dict[str, Any]


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


def _track_totals(clips: tuple[Clip, ...], tracks: dict[str, tuple[int, ...]]) -> tuple[int, int, int]:
    frames = sum(clip.n_frames for clip in clips)
    values = (tracks[clip.clip_id] for clip in clips)
    changes = sum(sum(a != b for a, b in zip(track, track[1:], strict=False)) for track in values)
    coast = sum(sum(abs(value) for value in tracks[clip.clip_id]) for clip in clips)
    return frames, changes, coast


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
) -> BudgetSeedRun:

    gate, train_frames = _train_count_gate(seed, train_clips, features_by_clip, gt_by_clip, config)
    total_frames = int(sum(clip.n_frames for clip in test_clips))

    val_probs = np.concatenate(
        [
            causal_gate_trace(gate, features_by_clip[clip.clip_id], 0.5, CountOnlineState.initial)[1]
            for clip in val_clips
        ]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    for rate in config.target_rates:
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
            candidate_r, _ = causal_gate_trace(gate, features, theta, CountOnlineState.initial)
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
        arm_scores = {
            kind: score_arm([(gt, estimator, r) for _, gt, estimator, r in rows], COLD_START).payload()
            for kind, rows in arm_rows.items()
        }
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
    noise_reestimates, _ = causal_gate_trace(gate, noise_features, operating_theta, CountOnlineState.initial)
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
) -> CountArtifact:
    config = config or RealCountBedConfig()
    featurizer, estimator = FrozenCountFeaturizer(), FrozenCountEstimator()
    adapter = RealStarssAdapter(foa_root, metadata_root, max_frames=config.max_frames)
    gt_by_clip = {clip.clip_id: adapter.count_track(clip.clip_id) for clip in adapter.clips()}
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    estimator_by_clip = map_clip_audio(adapter, estimator.estimate_track)
    split = native_fold_split(adapter, config.n_val_rooms, refusal=CountProducerRefusal)
    train_clips, val_clips, test_clips = split.train, split.val, split.test
    train_frames, train_changes, _ = _track_totals(train_clips, gt_by_clip)
    n_test_frames, n_test_changes, test_coast = _track_totals(test_clips, gt_by_clip)
    train_density = train_changes / train_frames
    test_coast_from_zero = test_coast / n_test_frames
    operating_rate = min(config.target_rates, key=lambda rate: abs(rate - train_density))
    if n_test_changes == 0:
        raise CountProducerRefusal("the real test split carries no count changes to track")
    prereg = build_count_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        n_test_clips=len(test_clips),
        n_test_changes=n_test_changes,
        n_test_frames=n_test_frames,
        train_change_density=train_density,
        coast_from_zero_mae=test_coast_from_zero,
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_mae = float(prereg["sesoi"]["sesoi_mae"])
    pooled = np.concatenate([features_by_clip[clip.clip_id] for clip in test_clips])
    target_mean, target_std = float(pooled.mean()), float(pooled.std())
    seed_runs = [
        _run_seed_real(
            seed,
            train_clips,
            val_clips,
            test_clips,
            features_by_clip,
            estimator_by_clip,
            gt_by_clip,
            _real_noisy_tv_features(seed, config.noisy_tv_frames, featurizer, target_mean, target_std),
            config,
            train_density,
        )
        for seed in config.seeds
    ]
    report = run_matched_budget(
        COUNT_BUDGET_POLICY,
        seed_runs,
        score_group="arm_scores",
        score_field="mae",
        action_group="reestimations",
        flop_model=lambda kind: _flop_model(
            kind, seed_runs[0].total_frames, seed_runs[0].train_frames, config
        ),
        operating_budget_id=seed_runs[0].operating_budget_id,
        source_kind="real",
        ceiling=FLOP_CEILING,
    )
    per_seed = [run.per_seed_block for run in seed_runs]
    deltas = [
        block["arm_scores"][ARM_RATE_MATCHED_RANDOM]["mae"] - block["arm_scores"][ARM_CANDIDATE]["mae"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    exceeds_sesoi = sesoi_exceeded(sign_flip.mean_delta, sesoi_mae)
    candidate_delta = -float(sign_flip.mean_delta)
    stats = count_sign_flip_payload(
        sign_flip,
        deltas,
        sesoi=sesoi_mae,
        exceeds_sesoi=exceeds_sesoi,
        mean_candidate_minus_control=candidate_delta,
        prereg_digest=prereg["canonical_sha256"],
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
    flags = {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    dominates = report.candidate_strictly_dominates_rate_matched_random
    meets_bar = dominates and sign_flip.one_sided_significant and exceeds_sesoi
    verdict = "mechanics-ok" if meets_bar else "null"
    truncations = [truncation.payload() for truncation in adapter.truncations()]
    core_evidence = {
        "per_seed": per_seed,
        "stats": stats,
        "controls": controls,
        "matched_budget": report.matched_budget.payload(),
        "flags": flags,
    }
    receipt = {
        "schema": "mop-ladder-run-receipt/v1",
        "kind": "mechanics-demonstration",
        "mechanism_id": COUNT_BED_ID,
        "stage": STAGE,
        "requirement_id": "stage3.confirmed_useful_mechanism",
        "verdict": verdict,
        "controls_cleared": [ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"],
        "evidence_digest": canonical_sha256(core_evidence),
        "overturns_null": "",
        "matched": None,
        "detail": {
            "source_kind": "real",
            "forcing_null": STAGE3_FORCING_NULL,
            "question": (
                "concurrent-source counting (distinct from the seven sealed onset-localization nulls)"
            ),
            "note": (
                "one real run is a mechanics demonstration; scientific confirmation needs the independent "
                "verifier plus at least three bias-independent reproductions and cannot be self-certified"
            ),
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
        },
        "claim_scope": COUNT_BUDGET_POLICY.claim_scope,
    }
    real_corpus = {
        "producer_schema": COUNT_PRODUCER_SCHEMA,
        "foa_root": str(Path(foa_root)),
        "metadata_root": str(Path(metadata_root)),
        "n_clips": len(adapter.clips()),
        "split_rooms": dict(split.detail),
        "n_train_clips": len(train_clips),
        "n_val_clips": len(val_clips),
        "n_train_frames": seed_runs[0].train_frames,
        "n_test_clips": len(test_clips),
        "n_test_frames": n_test_frames,
        "n_test_changes": n_test_changes,
        "train_change_density": round(float(train_density), 12),
        "test_coast_from_zero_mae": round(float(test_coast_from_zero), 12),
        "operating_reestimate_fraction": round(float(operating_rate), 12),
        "truncation": {
            "clips_capped_by_max_frames": sum(1 for item in truncations if item["capped_by_max_frames"]),
            "onsets_dropped_past_audio_end": sum(item["dropped_onsets_past_end"] for item in truncations),
            "max_frames": config.max_frames,
            "per_clip": truncations,
        },
    }
    corpus_tracks = {
        clip.clip_id: {
            "n_frames": clip.n_frames,
            "gt_count_track": list(gt_by_clip[clip.clip_id]),
            "estimator_track": [int(value) for value in estimator_by_clip[clip.clip_id].tolist()],
        }
        for clip in test_clips
    }
    body = {
        "schema": ARTIFACT_SCHEMA,
        "stage": STAGE,
        "bed_id": report.policy.bed_id,
        "claim_scope": report.policy.claim_scope,
        "source_kind": report.source_kind,
        "rights_clean": True,
        "reproductions": 0,
        "seeds": list(config.seeds),
        "per_seed": per_seed,
        "stats": stats,
        "controls": controls,
        "flags": flags,
        "verdict": verdict,
        "harness": report.payload(),
        "featurizer": {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
        },
        "gate": {
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
        "demonstration_receipt": receipt,
        "matched_budget": report.matched_budget.payload(),
        "matched_budget_wall_note": MATCHED_BUDGET_WALL_NOTE,
        "break_even": report.break_even.payload(),
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate,
        },
        "cold_start": COLD_START,
        "primary_control": ARM_RATE_MATCHED_RANDOM,
        "corpus_tracks": corpus_tracks,
        "estimator": {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        },
        "real_corpus": real_corpus,
        "prereg": {
            "path": str(prereg_written),
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_mae": sesoi_mae,
            "provisional": False,
            "written_before_test_scores": True,
        },
    }
    body["seal"] = canonical_sha256(body)
    return CountArtifact(body, prereg)
