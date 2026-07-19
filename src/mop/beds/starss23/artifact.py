
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    FlopModel,
    arm_flop_model,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.gating import assemble_causal_inputs, causal_gate_trace
from mop.science.statistics import (
    BOUNDED_CLAIM_VERB,
    PROVISIONAL_SESOI_F1,
    exact_sign_flip,
    sign_flip_payload,
)

from . import BED_ID, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import SyntheticStarssAdapter, metadata_text_from_onsets
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .experiments import ONSET_BUDGET_POLICY
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .fixtures import REGIME_FAVORABLE, REGIME_NULL, SyntheticStarssConfig, generate_clip
from .gate import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PONDER_LAMBDA,
    FLOPS_PER_INFERENCE,
    CandidateGate,
    OnlineState,
    training_flops,
)
from .referee import score_arm
from .schema import COLLAR_FRAMES, FRAME_MS

ARTIFACT_SCHEMA = "mop-starss23-escs-bed/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

DOWNSTREAM_FLOPS_PER_FIRING = 40_000

FULL_SCALE_TRAIN_FRAMES = 54_000
FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_C_TRAIN = training_flops(FULL_SCALE_TRAIN_FRAMES, DEFAULT_EPOCHS)  # ~8.27e9
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME * FULL_SCALE_TEST_FRAMES  # ~2.691e10


class ArtifactRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BedConfig:

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    clip_seconds: float = 8.0
    clips_per_room: int = 3
    n_train_rooms: int = 2
    n_val_rooms: int = 1
    n_test_rooms: int = 2
    onsets_per_clip: int = 6
    nuisance_per_clip: int = 7
    voc_window: int = 1
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    ponder_lambda: float = DEFAULT_PONDER_LAMBDA
    target_rates: tuple[float, ...] = (0.10, 0.08, 0.06)
    noisy_tv_frames: int = 1000
    downstream_flops_per_firing: int = DOWNSTREAM_FLOPS_PER_FIRING

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise ArtifactRefusal("the bed needs at least two paired seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ArtifactRefusal("paired seeds must be unique")
        if self.clips_per_room <= 0:
            raise ArtifactRefusal("clips_per_room must be positive")
        if not self.target_rates:
            raise ArtifactRefusal("at least one firing budget target rate is required")

    @property
    def n_rooms(self) -> int:
        return self.n_train_rooms + self.n_val_rooms + self.n_test_rooms


def _build_adapter(seed: int, config: BedConfig) -> SyntheticStarssAdapter:

    fixture_config = SyntheticStarssConfig(
        clip_seconds=config.clip_seconds,
        onsets_per_clip=config.onsets_per_clip,
        nuisance_per_clip=config.nuisance_per_clip,
        base_seed=seed,
    )
    audio_by_clip: dict[str, np.ndarray] = {}
    metadata_by_clip: dict[str, str] = {}
    for room in range(config.n_rooms):
        fold = 3 if room < config.n_train_rooms + config.n_val_rooms else 4
        for mix in range(config.clips_per_room):
            clip_id = f"fold{fold}_room{room}_mix{mix:03d}"
            clip, audio = generate_clip(
                clip_id=clip_id,
                room_id=f"room{room:02d}",
                regime=REGIME_FAVORABLE,
                config=fixture_config,
            )
            audio_by_clip[clip_id] = audio
            metadata_by_clip[clip_id] = metadata_text_from_onsets(clip.onsets)
    return SyntheticStarssAdapter(audio_by_clip, metadata_by_clip)


def _featurize(adapter: SyntheticStarssAdapter, featurizer: FrozenFeaturizer) -> dict[str, np.ndarray]:
    return {clip.clip_id: featurizer.featurize(adapter.audio(clip.clip_id)) for clip in adapter.clips()}


def _voc_targets(gt_frames: Sequence[int], n_frames: int, window: int = 1) -> np.ndarray:

    targets = np.zeros(n_frames, dtype=np.float64)
    for frame in range(n_frames):
        if any(abs(frame - gt) <= window for gt in gt_frames):
            targets[frame] = 1.0
    return targets


def _noisy_tv_channel(seed: int, config: BedConfig, featurizer: FrozenFeaturizer) -> np.ndarray:

    n_frames = config.noisy_tv_frames
    favorable_frames = max(1, round(config.clip_seconds * 1000.0 / FRAME_MS))
    density = config.nuisance_per_clip / favorable_frames
    null_config = SyntheticStarssConfig(
        clip_seconds=n_frames * FRAME_MS / 1000.0,
        onsets_per_clip=1,
        nuisance_per_clip=max(1, round(density * n_frames)),
        base_seed=int(seed),
    )
    _clip, audio = generate_clip(
        clip_id="fold3_room99_mix000", room_id="room99", regime=REGIME_NULL, config=null_config
    )
    return featurizer.featurize(audio)


def _train_gate(
    seed: int,
    split_train: Sequence,
    features_by_clip: dict[str, np.ndarray],
    config: BedConfig,
) -> tuple[CandidateGate, int]:

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in split_train:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, OnlineState.initial))
        targets.append(_voc_targets(clip.onset_frames, clip.n_frames, window=config.voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = CandidateGate(seed=seed)
    gate.fit(
        x,
        y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
    )
    return gate, int(x.shape[0])


@dataclass(frozen=True, slots=True)
class _SeedRun:
    seed: int
    total_frames: int
    train_frames: int
    gate_params: int
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    per_seed_block: dict[str, Any]
    noisy_tv: dict[str, Any]


def _pooled_score(clip_gt_and_fires: Sequence[tuple[list[int], list[int]]]) -> dict[str, Any]:
    return score_arm(clip_gt_and_fires, COLLAR_FRAMES).payload()


def _run_seed(seed: int, config: BedConfig, featurizer: FrozenFeaturizer) -> _SeedRun:
    adapter = _build_adapter(seed, config)
    features_by_clip = _featurize(adapter, featurizer)
    split = adapter.harness_split(n_train_rooms=config.n_train_rooms, n_val_rooms=config.n_val_rooms)

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
            candidate_fires, _ = causal_gate_trace(gate, features, theta, OnlineState.initial)
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

    onset_density = config.onsets_per_clip / max(1, round(config.clip_seconds * 1000.0 / FRAME_MS))
    operating_budget_id = min(
        per_budget, key=lambda bid: abs(per_budget[bid]["rate"] - onset_density)
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
    noise_features = _noisy_tv_channel(seed, config, featurizer)
    noise_fires, _ = causal_gate_trace(
        gate, noise_features, operating_theta, OnlineState.initial
    )
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": config.noisy_tv_frames,
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


def _flop_model(kind: str, total_frames: int, train_frames: int, config: BedConfig) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME,
        gate_infer_per_frame=FLOPS_PER_INFERENCE,
        downstream_flops_per_firing=config.downstream_flops_per_firing,
        candidate_train_flops=lambda: training_flops(train_frames, config.epochs),
    )


def build_bed_artifact(config: BedConfig | None = None) -> ArtifactResult:

    config = config or BedConfig()
    featurizer = FrozenFeaturizer()
    started = time.perf_counter_ns()
    seed_runs = [_run_seed(seed, config, featurizer) for seed in config.seeds]
    measured_wall_ns = max(1, time.perf_counter_ns() - started)

    budget_points = build_budget_points(
        ONSET_BUDGET_POLICY, seed_runs, score_group="arm_scores", score_field="f1",
        action_group="firings",
        flop_model=lambda kind: _flop_model(
            kind, seed_runs[0].total_frames, seed_runs[0].train_frames, config
        ),
    )
    nominal_wall_ns = max(1, max(point.candidate.max_lifecycle_flops() for point in budget_points))
    report = run_matched_budget(
        budget_points,
        wall_ns=nominal_wall_ns,
        operating_budget_id=seed_runs[0].operating_budget_id,
        source_kind="synthetic",
        ceiling=FLOP_CEILING,
    )

    per_seed = [run.per_seed_block for run in seed_runs]
    deltas = [
        block["arm_scores"][ARM_CANDIDATE]["f1"] - block["arm_scores"][PRIMARY_CONTROL]["f1"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    stats_block = sign_flip_payload(
        sign_flip, deltas, sesoi_key="sesoi_f1", sesoi=PROVISIONAL_SESOI_F1,
        exceeds_sesoi=sign_flip.mean_delta >= PROVISIONAL_SESOI_F1,
        claim_verb=BOUNDED_CLAIM_VERB,
    )

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["firing_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls_block = noise_control_summary(
        ONSET_BUDGET_POLICY, seed_runs, at_chance=noisy_tv_at_chance, mean_noise_rate=mean_noise_rate,
        mean_base_rate=mean_base_rate, rate_key="mean_firing_rate_on_noise",
    )
    flags_block = safety_flags()

    dominates = report.candidate_strictly_dominates_rate_matched_random
    meets_bar = dominates and sign_flip.one_sided_significant and stats_block["mean_delta_exceeds_sesoi"]
    verdict = VERDICT_MECHANICS_OK if meets_bar else VERDICT_NULL

    core_evidence = {
        "per_seed": per_seed,
        "stats": stats_block,
        "controls": controls_block,
        "matched_budget": report.matched_budget.payload(),
        "flags": flags_block,
    }
    receipt = demonstration_receipt(
        mechanism_id=BED_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv"),
        evidence=core_evidence,
        verdict=verdict,
        detail={
            "source_kind": "synthetic",
            "forcing_null": STAGE3_FORCING_NULL,
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "synthetic fixtures cannot clear a stage gate; this is a mechanics demonstration only"
            ),
        },
    )

    body = artifact_envelope(
        schema=ARTIFACT_SCHEMA, report=report, seeds=config.seeds, per_seed=per_seed,
        stats=stats_block, controls=controls_block, flags=flags_block, verdict=verdict,
        featurizer={
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        }, gate={
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        }, receipt_payload=receipt,
        extra={
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_firing": config.downstream_flops_per_firing,
                "break_even_frames_anchor": FULL_SCALE_C_TRAIN // config.downstream_flops_per_firing,
            },
        },
    )
    return finalize_artifact(
        body,
        receipt_payload=receipt,
        verdict=verdict,
        detail={
            "dominates": dominates,
            "mean_delta": float(sign_flip.mean_delta),
            "one_sided_p": float(sign_flip.one_sided_p),
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
        },
    )


DEFAULT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED.json")
