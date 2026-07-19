"""Real-data producer for the STARSS23 counting bed gate-architecture reproduction.

This is a net-new, ADDITIVE component. It runs the whole counting bed end to end on the REAL, MIT-licensed
STARSS23 FOA subset and assembles the byte-sealed ``proof/STARSS23_COUNTING_REPRO_gate_arch.json`` that the
separately-authored independent verifier re-scores from specification. It varies exactly ONE axis: the shape
of the trained gate (the sealed ``264 -> 12 -> 1`` becomes ``264 -> 8 -> 4 -> 1``). Every other stage is
held byte-identical to the sealed bed and reused BY REFERENCE:

- the native fold-respecting room-disjoint split, the one-time featurization and estimation, the label-free
  online-state input assembly, and the real noisy-TV channel are imported unchanged from ``count_producer``
  (they are all gate-agnostic, so importing them guarantees they cannot drift);
- the frozen count featurizer and estimator, the coasted-count-MAE referee, the four controls, the paired
  -seed exact sign-flip and SESOI statistics, the matched-budget harness and its FLOP ceiling, and the
  count-label derivation are all reused unchanged from their sealed modules.

Only the gate and its analytic FLOP and parameter anchors move, and the paired-seed family is the disjoint
gate-architecture family (40..44) so the reproduction shares none of the original's seed luck. The matched
-budget invariant still holds because the candidate and rate-matched-random arms are charged the SAME new
per-frame gate-inference cost, and the candidate alone charges the new architecture's C_train.

The SESOI is preregistered before any test score is read (see ``count_repro_gate_arch_prereg``); this
producer writes the sealed prereg first and records its digest in the artifact. The verdict is a mechanics
demonstration only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single reproduction can never be
scientifically confirmed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

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
    FlopModel,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.gating import assemble_causal_inputs
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_check
from mop.substrate.events import write_canonical_json

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split
from .controls import (
    at_chance,
)
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import CountOnlineState
from .count_labels import build_count_clips, change_density, coast_from_zero_mae

# Gate-agnostic pipeline stages reused by reference from the sealed producer so they cannot drift.
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    CountProducerRefusal,
    RealCountBedConfig,
    _micro_count_score,
    _real_noisy_tv_features,
    run_count_seed,
)
from .count_referee import COLD_START
from .count_repro_gate_arch_gate import (
    D_IN_GATE_ARCH,
    FLOPS_PER_INFERENCE_GATE_ARCH,
    HIDDEN1,
    HIDDEN2,
    N_OUT,
    REPRO_AXIS,
    CountReproGateArchGate,
    training_flops_two_layer,
    voc_targets_from_count_track,
)
from .count_repro_gate_arch_prereg import (
    COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
    DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
    build_count_repro_gate_arch_prereg,
)
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY
from .schema import Clip

COUNT_REPRO_GATE_ARCH_PRODUCER_SCHEMA = "mop-starss23-count-repro-gate-arch-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-gate-arch/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The disjoint gate-architecture seed family: no overlap with the original (0..4), so no shared seed luck.
GATE_ARCH_SEEDS: tuple[int, ...] = (40, 41, 42, 43, 44)


class CountReproGateArchProducerRefusal(ValueError):
    """Raised when the gate-architecture reproduction cannot assemble a well-formed sealed artifact."""


def _train_count_gate(
    seed: int,
    train_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    config: RealCountBedConfig,
) -> tuple[CountReproGateArchGate, int]:
    """Train the re-authored two-layer gate on train-room value-of-computation targets.

    The shared causal input assembly and ``voc_targets_from_count_track`` are held fixed; only the gate
    class is the varied axis.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, CountOnlineState.initial))
        targets.append(voc_targets_from_count_track(gt_by_clip[clip.clip_id], window=config.voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = CountReproGateArchGate(seed=seed)
    gate.fit(
        x,
        y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
    )
    return gate, int(x.shape[0])


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
    """Bind the alternate gate to the held-fixed counting seed lifecycle."""

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


def _flop_model(
    kind: str, total_frames: int, train_frames: int, config: RealCountBedConfig
) -> FlopModel:
    """Full-lifecycle FLOP model for one arm, charging the re-authored gate's per-frame and training costs.

    Both the candidate and the rate-matched-random control are charged the SAME new per-frame gate-inference
    cost, so their inference FLOPs stay byte-equal and the matched-budget invariant holds; the candidate
    alone charges the new architecture's amortized C_train.
    """

    featurize = FLOPS_PER_FRAME_COUNT * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = FLOPS_PER_INFERENCE_GATE_ARCH * total_frames if runs_gate else 0
    train = training_flops_two_layer(train_frames, config.epochs) if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        train_flops=train,
    )


# ---------------------------------------------------------------------------
# Assemble and seal the gate-architecture reproduction artifact.
# ---------------------------------------------------------------------------


def build_real_count_repro_gate_arch_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
) -> ArtifactResult:
    """Run the whole counting bed with the re-authored gate on the real STARSS23 subset and seal it.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by
    the caller and never read from the wall clock inside a sealed body.
    """

    config = config or RealCountBedConfig(seeds=GATE_ARCH_SEEDS)
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

    # Label-only structural facts for the SESOI and the operating-point rule. No test score is read here.
    train_count_clips = [count_clips[c.clip_id] for c in train_clips]
    test_count_clips = [count_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_count_clips)
    n_test_clips = len(test_clips)
    n_test_frames = int(sum(clip.n_frames for clip in test_clips))
    n_test_changes = int(sum(cc.n_changes for cc in test_count_clips))
    test_coast_from_zero = coast_from_zero_mae(test_count_clips)
    if n_test_changes == 0:
        raise CountReproGateArchProducerRefusal("the real test split carries no count changes to track")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_count_repro_gate_arch_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        n_test_frames=n_test_frames,
        train_change_density=train_density,
        coast_from_zero_mae=test_coast_from_zero,
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_mae = float(prereg["sesoi"]["sesoi_mae"])

    # 2. Now run the paired seeds and score the test split.
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
            _run_seed_real(
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
        COUNT_BUDGET_POLICY, seed_runs, score_group="arm_scores", score_field="mae",
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
    # delta_i = MAE_rate_matched_random(i) - MAE_candidate(i). Positive = candidate reduces error.
    deltas = [
        block["arm_scores"][PRIMARY_CONTROL]["mae"] - block["arm_scores"][ARM_CANDIDATE]["mae"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1=sesoi_mae, provisional=False)
    mean_delta_exceeds_sesoi = bool(sesoi.exceeds_sesoi)
    mean_delta_candidate_minus_random = -float(sign_flip.mean_delta)

    stats_block = count_sign_flip_payload(
        sign_flip, deltas, sesoi=sesoi_mae, exceeds_sesoi=mean_delta_exceeds_sesoi,
        mean_candidate_minus_control=mean_delta_candidate_minus_random,
        prereg_digest=prereg["canonical_sha256"],
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
    meets_bar = dominates and sign_flip.one_sided_significant and mean_delta_exceeds_sesoi
    verdict = VERDICT_MECHANICS_OK if meets_bar else VERDICT_NULL

    # Operating-point survive readout (the design's strict conjunction at the preregistered budget point).
    operating_budget_id = seed_runs[0].operating_budget_id
    operating_row = next(
        row
        for row in report.per_budget_candidate_vs_rate_matched_random
        if row["budget_id"] == operating_budget_id
    )
    candidate_op_mae = float(operating_row["candidate_mean_mae"])
    rmr_op_mae = float(operating_row["rate_matched_random_mean_mae"])
    operating_candidate_beats_rmr = bool(operating_row["candidate_strictly_beats_rate_matched_random"])
    survives = bool(
        operating_candidate_beats_rmr
        and mean_delta_exceeds_sesoi
        and sign_flip.one_sided_significant
    )

    core_evidence = {
        "per_seed": per_seed,
        "stats": stats_block,
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
                "gate-architecture reproduction of concurrent-source counting: does a two-hidden-layer gate "
                "beat rate-matched-random at matched budget"
            ),
            "candidate_strictly_dominates_rate_matched_random": dominates,
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
            "architecture": "two_hidden_layer_mlp",
            "topology": "264 -> 8 -> 4 -> 1",
            "d_in": D_IN_GATE_ARCH,
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "n_out": N_OUT,
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "parameter_digest_seed0": CountReproGateArchGate(seed=config.seeds[0]).parameter_digest(),
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE_GATE_ARCH,
            "sealed_gate_topology": "264 -> 12 -> 1",
            "sealed_gate_flops_per_inference": 6385,
        }, receipt_payload=receipt,
        extra={
        "reproduction_axis": REPRO_AXIS,
        "cold_start": COLD_START,
        "primary_control": PRIMARY_CONTROL,
        "corpus_tracks": corpus_tracks,
        "survives_operating_point": survives,
        "operating_point": {
            "budget_id": operating_budget_id,
            "candidate_mean_mae": candidate_op_mae,
            "rate_matched_random_mean_mae": rmr_op_mae,
            "candidate_strictly_beats_rate_matched_random": operating_candidate_beats_rmr,
        },
        "estimator": {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        },
        "real_corpus": {
            "producer_schema": COUNT_REPRO_GATE_ARCH_PRODUCER_SCHEMA,
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
            "schema": COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_mae": sesoi_mae,
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
            "dominates": dominates,
            "survives_operating_point": survives,
            "operating_budget_id": operating_budget_id,
            "candidate_operating_mae": candidate_op_mae,
            "rate_matched_random_operating_mae": rmr_op_mae,
            "mean_delta_control_minus_candidate": float(sign_flip.mean_delta),
            "mean_delta_candidate_minus_control": mean_delta_candidate_minus_random,
            "one_sided_p": float(sign_flip.one_sided_p),
            "one_sided_significant": bool(sign_flip.one_sided_significant),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_mae": sesoi_mae,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(v) for v in deltas],
        },
    )


DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_gate_arch.json")
