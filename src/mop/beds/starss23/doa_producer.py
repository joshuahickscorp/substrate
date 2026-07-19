"""Direction-of-arrival bed, component 9: the real-data producer, run under BOTH architectures.

This is a net-new, additive component. It runs the whole DoA bed end to end on the REAL, MIT-licensed
STARSS23 FOA subset served by ``RealStarssAdapter`` (the same fixed subset the counting bed used) and
assembles the byte-sealed ``proof/STARSS23_DOA_BED.json`` the separately authored independent verifier
re-scores from specification. It edits no sealed module: the paired-seed statistics (``stats.
exact_sign_flip``) and the rate-matched-random/always-on controls are imported unchanged; the DoA labels,
featurizer, estimator, gate (both architectures), referee, controls, harness, and preregistration are the
net-new DoA modules built alongside this one.

The corpus is fixed real data, so the adapter is built once, every clip is featurized once with the frozen
``DoaFeaturizer``, and the frozen ``FrozenDoaEstimator`` track is computed once per clip; only the trained
gate (per architecture, per seed) and the rate-matched-random permutation vary. The room-disjoint split
respects the native STARSS23 fold boundary exactly as the counting bed's producer does: test is fold-4
dev-test, val is carved from the tail of fold-3 dev-train rooms, train is the rest of fold-3.

The SESOI and the full SURVIVES(X) promotion rule are preregistered before any test score is read (see
``doa_prereg.py``); this producer writes the sealed prereg first and records its digest in the artifact.
For each architecture, SURVIVES(X) requires ALL of: the point-estimate strict inequality; the registered
SESOI cleared on the mean clip-level (seed-averaged) delta; the primary clip-level exact sign-flip p
clearing alpha=0.05; the 5-seed secondary sign-flip agreeing in direction; and (the calibration note's
added discipline) the room-majority collapse not contradicting at alpha=0.10. The bed-level verdict is
"mechanics-ok" only when BOTH architectures survive, "architecture-fragile" when exactly one does, else
"null". ``activation_allowed``, ``scientific_promotion``, and ``independent_scientific_confirmation`` are
hardcoded false: a single run can never be scientifically confirmed, and a positive here is flagged for
human review only.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
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
    safety_flags,
)
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    BudgetPoint,
    FlopModel,
    build_budget_points,
    run_dual_architecture,
)
from mop.science.gating import assemble_causal_inputs, causal_gate_trace
from mop.science.statistics import BOUNDED_CLAIM_VERB, exact_sign_flip, sesoi_check
from mop.substrate.events import write_canonical_json

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, map_clip_audio, marginal_matched_noise, native_fold_split
from .controls import always_on_fires, at_chance, never_update_reestimates, rate_matched_random_fires
from .doa_estimator import FLOPS_PER_REESTIMATE, FrozenDoaEstimator
from .doa_featurizer import D_FEAT_DOA, DoaFeaturizer
from .doa_featurizer import FLOPS_PER_FRAME as FEATURIZER_FLOPS_PER_FRAME
from .doa_gate import (
    ARCH_A_ID,
    ARCH_B_ID,
    ARCHITECTURES,
    C_TRAIN_ANCHOR_ARCH_A,
    C_TRAIN_ANCHOR_ARCH_B,
    FLOPS_PER_INFERENCE_ARCH_A,
    FLOPS_PER_INFERENCE_ARCH_B,
    DoaOnlineState,
    build_gate,
    training_flops_arch_a,
    training_flops_arch_b,
)
from .doa_labels import (
    build_doa_clips,
    change_density,
    doa_voc_targets_from_track,
    mean_change_jump_deg,
    to_arrays,
)
from .doa_prereg import (
    DEFAULT_DOA_PREREG_PATH,
    ROOM_MAJORITY_ALPHA,
    DoaClipLabelFact,
    build_doa_prereg,
)
from .doa_referee import (
    DOA_COLD_START,
    exact_sign_flip_over_clips,
    macro_score_arm,
    pooled_score_arm,
    room_majority_collapse,
)
from .experiments import DOA_BED_ID, DOA_BUDGET_POLICY
from .gate import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PONDER_LAMBDA
from .schema import Clip

DOA_PRODUCER_SCHEMA = "mop-starss23-doa-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-doa-bed/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The same fixed real STARSS23 FOA subset the counting bed used, so the two lanes are directly comparable.
DEFAULT_FOA_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
DEFAULT_METADATA_ROOT = Path(
    "/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev"
)

DEFAULT_N_VAL_ROOMS = 2
_ALL_KINDS: tuple[str, ...] = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE)
_GATE_TRAINED_KINDS: tuple[str, ...] = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)


class DoaProducerRefusal(ValueError):
    """Raised when the DoA producer cannot assemble a well-formed sealed artifact."""


@dataclass(frozen=True, slots=True)
class RealDoaBedConfig:
    """Real-run configuration. Five paired seeds per architecture; the corpus is fixed."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS
    target_rates: tuple[float, ...] = (0.10, 0.05, 0.02)
    noisy_tv_frames: int = 2000
    max_frames: int | None = None
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    ponder_lambda: float = DEFAULT_PONDER_LAMBDA
    voc_window: int = 1
    downstream_flops_per_reestimate: int = FLOPS_PER_REESTIMATE

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise DoaProducerRefusal("the bed needs at least two paired seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise DoaProducerRefusal("paired seeds must be unique")
        if not self.target_rates:
            raise DoaProducerRefusal("at least one re-estimation budget target rate is required")


def _train_gate(
    architecture: str,
    seed: int,
    train_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    doa_track_by_clip: dict[str, tuple],
    config: RealDoaBedConfig,
) -> tuple[Any, int]:
    """Train one candidate gate of the named architecture on train-room value-of-computation targets."""

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, DoaOnlineState.initial))
        targets.append(doa_voc_targets_from_track(doa_track_by_clip[clip.clip_id], window=config.voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = build_gate(architecture, seed=seed)
    gate.fit(
        x, y, epochs=config.epochs, learning_rate=config.learning_rate, ponder_lambda=config.ponder_lambda
    )
    return gate, int(x.shape[0])


# ---------------------------------------------------------------------------
# The real noisy-TV channel: white-noise audio featurized and marginal-matched to the real test content.
# ---------------------------------------------------------------------------


def _noise_seed(seed: int) -> int:
    payload = json.dumps(
        {"seed": int(seed), "key": "mop.beds.starss23.doa.noisy_tv"},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b"mop-starss23-doa-noisy-tv-v1\0" + payload).digest()[:4], "big")


def _real_noisy_tv_features(
    seed: int, n_frames: int, featurizer: DoaFeaturizer, target_mean: float, target_std: float
) -> np.ndarray:
    """Build the DoA bed's independently seeded aleatoric control channel."""

    return marginal_matched_noise(_noise_seed(seed), n_frames, featurizer, target_mean, target_std)


# ---------------------------------------------------------------------------
# Architecture-independent arms (always_on, never_update): computed ONCE, reused everywhere.
# ---------------------------------------------------------------------------


def _deterministic_arm_tuples(
    kind: str, test_clips: tuple[Clip, ...], arrays_by_clip: dict[str, tuple[np.ndarray, np.ndarray]],
    estimator_by_clip: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray, np.ndarray, list[int], np.ndarray]]:
    tuples = []
    for clip in test_clips:
        active_mask, gt_directions = arrays_by_clip[clip.clip_id]
        r = (
            always_on_fires(clip.n_frames)
            if kind == ARM_ALWAYS_ON
            else never_update_reestimates(clip.n_frames)
        )
        tuples.append((clip.clip_id, gt_directions, estimator_by_clip[clip.clip_id], r, active_mask))
    return tuples


# ---------------------------------------------------------------------------
# Per-seed, per-architecture run on the fixed real test split.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SeedRun:
    seed: int
    architecture: str
    total_frames: int
    train_frames: int
    gate_params: int
    parameter_digest: str
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    noisy_tv: dict[str, Any]


def _run_seed_real(
    architecture: str,
    seed: int,
    train_clips: tuple[Clip, ...],
    val_clips: tuple[Clip, ...],
    test_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    estimator_by_clip: dict[str, np.ndarray],
    doa_track_by_clip: dict[str, tuple],
    arrays_by_clip: dict[str, tuple[np.ndarray, np.ndarray]],
    noise_features: np.ndarray,
    config: RealDoaBedConfig,
    operating_density: float,
    always_on_payloads: dict[str, dict[str, Any]],
    never_update_payloads: dict[str, dict[str, Any]],
) -> _SeedRun:
    """Train the gate for one (architecture, seed); sweep the budget, score every arm on the test set."""

    gate, train_frames = _train_gate(
        architecture, seed, train_clips, features_by_clip, doa_track_by_clip, config
    )
    total_frames = int(sum(clip.n_frames for clip in test_clips))

    val_probs = np.concatenate(
        [causal_gate_trace(gate, features_by_clip[clip.clip_id], 0.5, DoaOnlineState.initial)[1]
         for clip in val_clips]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    for rate in config.target_rates:
        theta = float(np.quantile(val_probs, 1.0 - rate))
        budget_id = f"rate_{rate:.2f}"

        candidate_tuples: list[tuple[str, np.ndarray, np.ndarray, list[int], np.ndarray]] = []
        rmr_tuples: list[tuple[str, np.ndarray, np.ndarray, list[int], np.ndarray]] = []
        reestimation_counts = {ARM_CANDIDATE: 0, ARM_RATE_MATCHED_RANDOM: 0}
        reestimate_frames_by_clip: dict[str, dict[str, list[int]]] = {}
        for clip in test_clips:
            features = features_by_clip[clip.clip_id]
            active_mask, gt_directions = arrays_by_clip[clip.clip_id]
            estimator_track = estimator_by_clip[clip.clip_id]
            candidate_r, _ = causal_gate_trace(
                gate, features, theta, DoaOnlineState.initial
            )
            rmr_r = rate_matched_random_fires(candidate_r, clip.n_frames, seed=seed, clip_id=clip.clip_id)
            candidate_tuples.append(
                (clip.clip_id, gt_directions, estimator_track, list(candidate_r), active_mask)
            )
            rmr_tuples.append((clip.clip_id, gt_directions, estimator_track, list(rmr_r), active_mask))
            reestimation_counts[ARM_CANDIDATE] += len(candidate_r)
            reestimation_counts[ARM_RATE_MATCHED_RANDOM] += len(rmr_r)
            reestimate_frames_by_clip[clip.clip_id] = {
                ARM_CANDIDATE: list(candidate_r),
                ARM_RATE_MATCHED_RANDOM: list(rmr_r),
            }

        arm_scores_macro = {
            ARM_CANDIDATE: macro_score_arm(candidate_tuples).payload(),
            ARM_RATE_MATCHED_RANDOM: macro_score_arm(rmr_tuples).payload(),
            ARM_ALWAYS_ON: always_on_payloads["macro"],
            ARM_NEVER_UPDATE: never_update_payloads["macro"],
        }
        arm_scores_pooled = {
            ARM_CANDIDATE: pooled_score_arm(candidate_tuples).payload(),
            ARM_RATE_MATCHED_RANDOM: pooled_score_arm(rmr_tuples).payload(),
            ARM_ALWAYS_ON: always_on_payloads["pooled"],
            ARM_NEVER_UPDATE: never_update_payloads["pooled"],
        }
        reestimation_counts[ARM_ALWAYS_ON] = always_on_payloads["n_reestimations"]
        reestimation_counts[ARM_NEVER_UPDATE] = never_update_payloads["n_reestimations"]

        per_budget[budget_id] = {
            "theta": theta,
            "rate": rate,
            "reestimate_frames": reestimate_frames_by_clip,
            "arm_scores_macro": arm_scores_macro,
            "arm_scores_pooled": arm_scores_pooled,
            "reestimations": reestimation_counts,
        }

    operating_budget_id = min(per_budget, key=lambda bid: abs(per_budget[bid]["rate"] - operating_density))
    operating = per_budget[operating_budget_id]
    operating_theta = operating["theta"]
    base_rate = operating["reestimations"][ARM_CANDIDATE] / max(1, total_frames)
    noise_reestimates, _ = causal_gate_trace(
        gate, noise_features, operating_theta, DoaOnlineState.initial
    )
    noise_rate = len(noise_reestimates) / noise_features.shape[0]
    noisy_tv = {
        "reestimate_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    return _SeedRun(
        seed=seed,
        architecture=architecture,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        parameter_digest=gate.parameter_digest(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        noisy_tv=noisy_tv,
    )


# ---------------------------------------------------------------------------
# Harness arms across seeds, per architecture.
# ---------------------------------------------------------------------------


def _flop_model(
    kind: str, architecture: str, total_frames: int, train_frames: int, config: RealDoaBedConfig
) -> FlopModel:
    featurize = FEATURIZER_FLOPS_PER_FRAME * total_frames
    runs_gate = kind in _GATE_TRAINED_KINDS
    infer_per_frame = FLOPS_PER_INFERENCE_ARCH_A if architecture == ARCH_A_ID else FLOPS_PER_INFERENCE_ARCH_B
    gate_infer = infer_per_frame * total_frames if runs_gate else 0
    if kind == ARM_CANDIDATE:
        train_fn = training_flops_arch_a if architecture == ARCH_A_ID else training_flops_arch_b
        train = train_fn(train_frames, config.epochs)
    else:
        train = 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        train_flops=train,
    )


# ---------------------------------------------------------------------------
# Per-architecture statistical assembly: clip-level primary, 5-seed secondary, room-majority, SESOI.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ArchitectureStats:
    candidate_mean_macro_mae_deg: float
    rate_matched_random_mean_macro_mae_deg: float
    point_estimate_holds: bool
    per_clip_deltas: list[float]
    primary: Any  # ClipSignFlipResult
    secondary: Any  # SignFlipResult
    room_majority: Any  # RoomMajorityResult
    sesoi: Any  # SesoiCheck
    survives: bool
    pooled_mean_delta_deg: float  # secondary corroborating readout only, never the survive criterion


def _architecture_stats(
    arch_seed_runs: list[_SeedRun], test_clips: tuple[Clip, ...], sesoi_deg: float
) -> _ArchitectureStats:
    operating_scores = [
        run.per_budget[run.operating_budget_id]["arm_scores_macro"] for run in arch_seed_runs
    ]
    candidate_means = [scores[ARM_CANDIDATE]["macro_mae_deg"] for scores in operating_scores]
    rmr_means = [scores[ARM_RATE_MATCHED_RANDOM]["macro_mae_deg"] for scores in operating_scores]
    candidate_mean = math.fsum(candidate_means) / len(candidate_means)
    rmr_mean = math.fsum(rmr_means) / len(rmr_means)
    point_estimate_holds = candidate_mean < rmr_mean

    # 5-seed secondary sign-flip on the per-seed clip-macro deltas (rate_matched_random minus candidate).
    seed_level_deltas = [rmr - cand for cand, rmr in zip(candidate_means, rmr_means, strict=True)]
    secondary = exact_sign_flip(seed_level_deltas)

    # Primary clip-level sign-flip on per-clip deltas, each clip's own value averaged over the paired seeds.
    room_of: dict[str, str] = {clip.clip_id: clip.room_id for clip in test_clips}
    per_clip_deltas: list[float] = []
    by_room: dict[str, list[float]] = {}
    for clip in test_clips:
        clip_id = clip.clip_id
        vals = [
            scores[ARM_RATE_MATCHED_RANDOM]["per_clip"][clip_id]["mae_deg"]
            - scores[ARM_CANDIDATE]["per_clip"][clip_id]["mae_deg"]
            for scores in operating_scores
        ]
        delta = math.fsum(vals) / len(vals)
        per_clip_deltas.append(delta)
        by_room.setdefault(room_of[clip_id], []).append(delta)
    primary = exact_sign_flip_over_clips(per_clip_deltas)
    room_majority = room_majority_collapse(by_room)

    sesoi = sesoi_check(primary.mean_delta, sesoi_f1=sesoi_deg, provisional=False)

    survives = bool(
        point_estimate_holds
        and sesoi.exceeds_sesoi
        and primary.one_sided_significant
        and secondary.mean_delta > 0.0
        and room_majority.one_sided_p <= ROOM_MAJORITY_ALPHA
    )

    # SECONDARY, pooled-frame corroboration only: never the survive criterion, reported for comparison.
    pooled_deltas = [
        run.per_budget[run.operating_budget_id]["arm_scores_pooled"][ARM_RATE_MATCHED_RANDOM]["pooled_mae_deg"]
        - run.per_budget[run.operating_budget_id]["arm_scores_pooled"][ARM_CANDIDATE]["pooled_mae_deg"]
        for run in arch_seed_runs
    ]
    pooled_mean_delta_deg = math.fsum(pooled_deltas) / len(pooled_deltas)

    return _ArchitectureStats(
        candidate_mean_macro_mae_deg=candidate_mean,
        rate_matched_random_mean_macro_mae_deg=rmr_mean,
        point_estimate_holds=point_estimate_holds,
        per_clip_deltas=per_clip_deltas,
        primary=primary,
        secondary=secondary,
        room_majority=room_majority,
        sesoi=sesoi,
        survives=survives,
        pooled_mean_delta_deg=pooled_mean_delta_deg,
    )


# ---------------------------------------------------------------------------
# Assemble and seal the real DoA artifact.
# ---------------------------------------------------------------------------


def build_real_doa_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealDoaBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_DOA_PREREG_PATH,
) -> ArtifactResult:
    """Run the whole DoA bed on the real STARSS23 subset, under BOTH architectures, and seal the artifact."""

    config = config or RealDoaBedConfig()
    featurizer = DoaFeaturizer()
    estimator = FrozenDoaEstimator()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    doa_clips = build_doa_clips(adapter, metadata_root)
    doa_track_by_clip = {cid: dc.doa_track for cid, dc in doa_clips.items()}

    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    estimator_by_clip = map_clip_audio(adapter, estimator.estimate_track)
    split = native_fold_split(adapter, config.n_val_rooms, refusal=DoaProducerRefusal)
    train_clips, val_clips, test_clips = split.train, split.val, split.test
    split_detail = dict(split.detail)

    # Structural, label-only facts. No test score is read to build the prereg.
    train_doa_clips = [doa_clips[c.clip_id] for c in train_clips]
    test_doa_clips = [doa_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_doa_clips)
    jump_deg = mean_change_jump_deg(test_doa_clips)
    arrays_by_clip = {c.clip_id: to_arrays(doa_clips[c.clip_id]) for c in test_clips}

    test_clip_facts = [
        DoaClipLabelFact(clip_id=dc.clip_id, n_active_frames=dc.n_active_frames, n_changes=dc.n_changes)
        for dc in test_doa_clips
    ]
    if any(fact.n_changes for fact in test_clip_facts) is False:
        raise DoaProducerRefusal("the real test split carries no DoA changes to track")
    test_rooms = sorted({c.room_id for c in test_clips})
    n_test_active_frames = int(sum(dc.n_active_frames for dc in test_doa_clips))
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI, the promotion rule, and the analysis plan BEFORE reading any test score.
    prereg = build_doa_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        test_clip_facts=test_clip_facts,
        test_rooms=test_rooms,
        train_change_density=train_density,
        mean_change_jump_deg=jump_deg,
        c_train_flops_arch_a=C_TRAIN_ANCHOR_ARCH_A,
        c_train_flops_arch_b=C_TRAIN_ANCHOR_ARCH_B,
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_deg = float(prereg["sesoi"]["sesoi_deg"])

    # 2. Architecture-independent arms (always_on, never_update): computed ONCE, reused everywhere.
    always_on_tuples = _deterministic_arm_tuples(ARM_ALWAYS_ON, test_clips, arrays_by_clip, estimator_by_clip)
    never_update_tuples = _deterministic_arm_tuples(
        ARM_NEVER_UPDATE, test_clips, arrays_by_clip, estimator_by_clip
    )
    always_on_macro = macro_score_arm(always_on_tuples)
    always_on_pooled = pooled_score_arm(always_on_tuples)
    never_update_macro = macro_score_arm(never_update_tuples)
    never_update_pooled = pooled_score_arm(never_update_tuples)
    always_on_payloads = {
        "macro": always_on_macro.payload(),
        "pooled": always_on_pooled.payload(),
        "n_reestimations": int(sum(clip.n_frames for clip in test_clips)),
    }
    never_update_payloads = {
        "macro": never_update_macro.payload(),
        "pooled": never_update_pooled.payload(),
        "n_reestimations": 0,
    }

    # 3. Noisy-TV channel per seed, architecture-independent (raw audio through the frozen featurizer only).
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in test_clips], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())
    noise_features_by_seed = {
        seed: _real_noisy_tv_features(seed, config.noisy_tv_frames, featurizer, target_mean, target_std)
        for seed in config.seeds
    }

    # 4. Now run the paired seeds and score the test split, under BOTH architectures.
    started = time.perf_counter_ns()
    seed_runs_by_arch: dict[str, list[_SeedRun]] = {ARCH_A_ID: [], ARCH_B_ID: []}
    for architecture in ARCHITECTURES:
        for seed in config.seeds:
            seed_runs_by_arch[architecture].append(
                _run_seed_real(
                    architecture,
                    seed,
                    train_clips,
                    val_clips,
                    test_clips,
                    features_by_clip,
                    estimator_by_clip,
                    doa_track_by_clip,
                    arrays_by_clip,
                    noise_features_by_seed[seed],
                    config,
                    train_density,
                    always_on_payloads,
                    never_update_payloads,
                )
            )
    measured_wall_ns = max(1, time.perf_counter_ns() - started)

    def project_budget_points(architecture: str) -> list[BudgetPoint]:
        runs = seed_runs_by_arch[architecture]
        return build_budget_points(
            DOA_BUDGET_POLICY, runs, score_group="arm_scores_macro", score_field="macro_mae_deg",
            action_group="reestimations", architecture=architecture,
            flop_model=lambda kind: _flop_model(
                kind, architecture, runs[0].total_frames, runs[0].train_frames, config
            ),
        )

    budget_points_a = project_budget_points(ARCH_A_ID)
    budget_points_b = project_budget_points(ARCH_B_ID)
    nominal_wall_ns = max(
        1,
        max(point.candidate.max_lifecycle_flops() for point in (*budget_points_a, *budget_points_b)),
    )
    report = run_dual_architecture(
        budget_points_a,
        budget_points_b,
        wall_ns=nominal_wall_ns,
        source_kind="real",
        ceiling=FLOP_CEILING,
        operating_budget_id_a=seed_runs_by_arch[ARCH_A_ID][0].operating_budget_id,
        operating_budget_id_b=seed_runs_by_arch[ARCH_B_ID][0].operating_budget_id,
    )

    # 5. The full SURVIVES(X) statistical conjunction per architecture.
    stats_a = _architecture_stats(seed_runs_by_arch[ARCH_A_ID], test_clips, sesoi_deg)
    stats_b = _architecture_stats(seed_runs_by_arch[ARCH_B_ID], test_clips, sesoi_deg)
    both_survive = bool(stats_a.survives and stats_b.survives)
    exactly_one_survives = bool(stats_a.survives != stats_b.survives)
    if both_survive:
        bed_verdict = VERDICT_MECHANICS_OK
    elif exactly_one_survives:
        bed_verdict = "architecture-fragile"
    else:
        bed_verdict = VERDICT_NULL

    def _stats_payload(stats: _ArchitectureStats) -> dict[str, Any]:
        return {
            "candidate_mean_macro_mae_deg": round(stats.candidate_mean_macro_mae_deg, 12),
            "rate_matched_random_mean_macro_mae_deg": round(stats.rate_matched_random_mean_macro_mae_deg, 12),
            "point_estimate_holds": stats.point_estimate_holds,
            "per_clip_deltas": [round(v, 12) for v in stats.per_clip_deltas],
            "primary_clip_level_sign_flip": stats.primary.payload(),
            "secondary_seed_sign_flip": stats.secondary.payload(),
            "room_majority": stats.room_majority.payload(),
            "sesoi": stats.sesoi.payload(),
            "survives": stats.survives,
            "pooled_secondary_mean_delta_deg": round(stats.pooled_mean_delta_deg, 12),
            "claim_verb": BOUNDED_CLAIM_VERB,
        }

    stats_block = {
        "sesoi_deg": sesoi_deg,
        "prereg_canonical_sha256": prereg["canonical_sha256"],
        ARCH_A_ID: _stats_payload(stats_a),
        ARCH_B_ID: _stats_payload(stats_b),
        "both_architectures_survive": both_survive,
        "architecture_fragile": exactly_one_survives,
    }

    # 6. Noisy-TV at-chance, per architecture, mean over seeds.
    def _noisy_tv_block(seed_runs: list[_SeedRun]) -> dict[str, Any]:
        n_runs = len(seed_runs)
        mean_noise_rate = math.fsum(run.noisy_tv["reestimate_rate_on_noise"] for run in seed_runs) / n_runs
        mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
        return {
            "at_chance": at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate)),
            "mean_reestimate_rate_on_noise": round(float(mean_noise_rate), 12),
            "mean_base_rate": round(float(mean_base_rate), 12),
            "per_seed": [run.noisy_tv for run in seed_runs],
        }

    controls_block = {
        "primary_control": PRIMARY_CONTROL,
        "control_arms": [ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"],
        ARCH_A_ID: _noisy_tv_block(seed_runs_by_arch[ARCH_A_ID]),
        ARCH_B_ID: _noisy_tv_block(seed_runs_by_arch[ARCH_B_ID]),
    }
    noisy_tv_at_chance_both = bool(
        controls_block[ARCH_A_ID]["at_chance"] and controls_block[ARCH_B_ID]["at_chance"]
    )

    flags_block = safety_flags()

    # Shared per-clip ground-truth and estimator tracks, sealed once (identical across seeds and archs), so
    # the verifier can re-derive every arm, every seed, every architecture without re-running any DSP.
    corpus_tracks = {
        clip.clip_id: {
            "n_frames": clip.n_frames,
            "room_id": clip.room_id,
            "gt_track": [
                None if e is None else [float(e[0]), float(e[1])] for e in doa_clips[clip.clip_id].doa_track
            ],
            "estimator_track": [[float(a), float(b)] for a, b in estimator_by_clip[clip.clip_id].tolist()],
        }
        for clip in test_clips
    }

    def _per_seed_blocks(seed_runs: list[_SeedRun]) -> list[dict[str, Any]]:
        blocks = []
        for run in seed_runs:
            operating = run.per_budget[run.operating_budget_id]
            blocks.append(
                {
                    "seed": run.seed,
                    "operating_budget_id": run.operating_budget_id,
                    "gate_params": run.gate_params,
                    "parameter_digest": run.parameter_digest,
                    "reestimate_frames": operating["reestimate_frames"],
                    "arm_scores_macro": operating["arm_scores_macro"],
                    "arm_scores_pooled": operating["arm_scores_pooled"],
                    "reestimations": operating["reestimations"],
                    "per_budget_summary": {
                        bid: {
                            "rate": block["rate"],
                            "theta": block["theta"],
                            "candidate_macro_mae_deg": (
                                block["arm_scores_macro"][ARM_CANDIDATE]["macro_mae_deg"]
                            ),
                            "rate_matched_random_macro_mae_deg": (
                                block["arm_scores_macro"][ARM_RATE_MATCHED_RANDOM]["macro_mae_deg"]
                            ),
                        }
                        for bid, block in run.per_budget.items()
                    },
                }
            )
        return blocks

    per_seed_block = {
        ARCH_A_ID: _per_seed_blocks(seed_runs_by_arch[ARCH_A_ID]),
        ARCH_B_ID: _per_seed_blocks(seed_runs_by_arch[ARCH_B_ID]),
    }

    core_evidence = {
        "per_seed": per_seed_block,
        "stats": stats_block,
        "controls": controls_block,
        "harness": report.payload(),
        "flags": flags_block,
    }
    receipt_verdict = VERDICT_MECHANICS_OK if both_survive else VERDICT_NULL
    receipt = demonstration_receipt(
        mechanism_id=DOA_BED_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"),
        evidence=core_evidence,
        verdict=receipt_verdict,
        detail={
            "source_kind": "real",
            "forcing_null": STAGE3_FORCING_NULL,
            "question": "direction-of-arrival re-estimation (distinct from onset-localization and counting)",
            "bed_verdict": bed_verdict,
            "both_architectures_survive": both_survive,
            "architecture_fragile": exactly_one_survives,
            "note": (
                "one real run under both architectures is a mechanics demonstration; scientific "
                "confirmation needs the independent verifier plus at least three bias-independent "
                "reproductions and cannot be self-certified"
            ),
        },
    )

    truncations = [t.payload() for t in adapter.truncations()]

    body = artifact_envelope(
        schema=ARTIFACT_SCHEMA,
        report=report,
        seeds=config.seeds,
        per_seed=per_seed_block,
        stats=stats_block,
        controls=controls_block,
        flags=flags_block,
        verdict=bed_verdict,
        featurizer={
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FEATURIZER_FLOPS_PER_FRAME,
            "d_feat_doa": D_FEAT_DOA,
        },
        gate={
            ARCH_A_ID: {
                "topology": "264 -> 12 -> 1",
                "params": seed_runs_by_arch[ARCH_A_ID][0].gate_params,
                "param_ceiling": 4096,
                "flops_per_inference": FLOPS_PER_INFERENCE_ARCH_A,
                "c_train_anchor_flops": C_TRAIN_ANCHOR_ARCH_A,
            },
            ARCH_B_ID: {
                "topology": "264 -> 6 -> 6 -> 1",
                "params": seed_runs_by_arch[ARCH_B_ID][0].gate_params,
                "param_ceiling": 4096,
                "flops_per_inference": FLOPS_PER_INFERENCE_ARCH_B,
                "c_train_anchor_flops": C_TRAIN_ANCHOR_ARCH_B,
            },
            "state_bytes": DoaOnlineState.state_bytes(),
        },
        receipt_payload=receipt,
        extra={
            "cold_start": list(DOA_COLD_START),
            "architectures": list(ARCHITECTURES),
            "corpus_tracks": corpus_tracks,
            "estimator": {
                "n_params": estimator.n_params(),
                "parameter_digest": estimator.parameter_digest(),
                "flops_per_reestimate": FLOPS_PER_REESTIMATE,
            },
            "real_corpus": {
                "producer_schema": DOA_PRODUCER_SCHEMA,
                "foa_root": str(Path(foa_root)),
                "metadata_root": str(Path(metadata_root)),
                "n_clips": len(adapter.clips()),
                "split_rooms": split_detail,
                "n_train_clips": len(train_clips),
                "n_val_clips": len(val_clips),
                "n_train_frames": seed_runs_by_arch[ARCH_A_ID][0].train_frames,
                "n_test_clips": len(test_clips),
                "n_test_frames": int(sum(clip.n_frames for clip in test_clips)),
                "n_test_active_frames": n_test_active_frames,
                "n_test_changes": int(sum(dc.n_changes for dc in test_doa_clips)),
                "test_rooms": test_rooms,
                "train_change_density": round(float(train_density), 12),
                "mean_change_jump_deg": round(float(jump_deg), 12),
                "operating_reestimate_fraction": round(float(operating_rate), 12),
                "truncation": {
                    "clips_capped_by_max_frames": sum(
                        1 for t in truncations if t["capped_by_max_frames"]
                    ),
                    "onsets_dropped_past_audio_end": sum(
                        t["dropped_onsets_past_end"] for t in truncations
                    ),
                    "max_frames": config.max_frames,
                },
            },
            "prereg": {
                "path": str(prereg_written),
                "canonical_sha256": prereg["canonical_sha256"],
                "sesoi_deg": sesoi_deg,
                "provisional": False,
                "written_before_test_scores": True,
            },
            "noisy_tv_at_chance_both_architectures": noisy_tv_at_chance_both,
        },
    )
    return finalize_artifact(
        body,
        prereg=prereg,
        verdict=bed_verdict,
        detail={
            "both_architectures_survive": both_survive,
            "architecture_fragile": exactly_one_survives,
            ARCH_A_ID: {
                "survives": stats_a.survives,
                "candidate_mean_macro_mae_deg": stats_a.candidate_mean_macro_mae_deg,
                "rate_matched_random_mean_macro_mae_deg": stats_a.rate_matched_random_mean_macro_mae_deg,
                "mean_clip_delta_deg": stats_a.primary.mean_delta,
                "clip_level_one_sided_p": stats_a.primary.one_sided_p,
                "pooled_secondary_delta_deg": stats_a.pooled_mean_delta_deg,
            },
            ARCH_B_ID: {
                "survives": stats_b.survives,
                "candidate_mean_macro_mae_deg": stats_b.candidate_mean_macro_mae_deg,
                "rate_matched_random_mean_macro_mae_deg": stats_b.rate_matched_random_mean_macro_mae_deg,
                "mean_clip_delta_deg": stats_b.primary.mean_delta,
                "clip_level_one_sided_p": stats_b.primary.one_sided_p,
                "pooled_secondary_delta_deg": stats_b.pooled_mean_delta_deg,
            },
            "sesoi_deg": sesoi_deg,
            "noisy_tv_at_chance_both_architectures": noisy_tv_at_chance_both,
            "measured_wall_ns": measured_wall_ns,
            "max_lifecycle_flops_arch_a": max(
                point.candidate.max_lifecycle_flops() for point in budget_points_a
            ),
            "max_lifecycle_flops_arch_b": max(
                point.candidate.max_lifecycle_flops() for point in budget_points_b
            ),
        },
    )


DEFAULT_DOA_ARTIFACT_PATH = Path("proof/STARSS23_DOA_BED.json")
