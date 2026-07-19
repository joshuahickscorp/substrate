"""Real-data producer for the STARSS23 concurrent-source-counting value-of-computation bed.

This is a net-new, additive component. It runs the whole counting bed end to end on the REAL,
MIT-licensed STARSS23 FOA subset served by ``RealStarssAdapter`` and assembles the byte-sealed
``proof/STARSS23_COUNTING_BED.json`` the separately authored independent verifier re-scores from
specification. It edits no sealed onset scoring path: the paired-seed statistics
(``stats.exact_sign_flip``, ``stats.sesoi_check``) and the rate-matched-random and always-on controls are
imported unchanged; the count referee, the count harness FLOP accounting, the frozen count featurizer and
estimator, the count gate, and the count preregistration are the net-new counting modules.

The corpus is fixed real data, so the adapter is built once, every clip is featurized once, and the frozen
estimator track is computed once per clip; only the trained gate and the rate-matched-random permutation
vary across the five paired seeds. The room-disjoint split respects the native STARSS23 fold boundary: the
score partition is exactly the fold-4 dev-test rooms, and the val rooms are carved from the fold-3
dev-train rooms, so train, val, and test are room-disjoint and clip-disjoint.

The SESOI is preregistered before any test score is read (see ``count_prereg.py``); this producer writes
the sealed prereg first and records its digest in the artifact. The verdict is a mechanics demonstration
only: ``activation_allowed``, ``scientific_promotion``, and ``independent_scientific_confirmation`` are
hardcoded false, and a single run can never be scientifically confirmed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
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
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    BudgetSeedRun,
    FlopModel,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_check
from mop.substrate.events import write_canonical_json

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split
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
    FLOPS_PER_INFERENCE,
    CountGate,
    CountOnlineState,
    voc_targets_from_count_track,
)
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_prereg import (
    DEFAULT_COUNT_PREREG_PATH,
    build_count_prereg,
)
from .count_referee import COLD_START, score_arm
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY
from .gate import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PONDER_LAMBDA, training_flops
from .schema import N_CHANNELS, SAMPLES_PER_FRAME, Clip

COUNT_PRODUCER_SCHEMA = "mop-starss23-count-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The real STARSS23 FOA subset and metadata roots on this host (identical to the onset real producer).
DEFAULT_FOA_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
DEFAULT_METADATA_ROOT = Path(
    "/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev"
)

DEFAULT_N_VAL_ROOMS = 2

# Full-scale anchors from the recipe, recorded for provenance even when the fixed real subset is smaller.
FULL_SCALE_TRAIN_FRAMES = 54_000
FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_C_TRAIN = training_flops(FULL_SCALE_TRAIN_FRAMES, DEFAULT_EPOCHS)  # ~8.27e9
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME_COUNT * FULL_SCALE_TEST_FRAMES


class CountProducerRefusal(ValueError):
    """Raised when the count producer cannot assemble a well-formed sealed artifact."""


@dataclass(frozen=True, slots=True)
class RealCountBedConfig:
    """Real-run configuration. The paired seeds and the sweep mirror the recipe; the data is fixed."""

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


# ---------------------------------------------------------------------------
# Label-free online-state assembly and causal re-estimation passes.
# ---------------------------------------------------------------------------


def _assemble_inputs(features: np.ndarray) -> np.ndarray:
    """Assemble (n_frames, D_IN) gate inputs with a label-free causal online-state pass.

    The online state is advanced with no re-estimation (self-derived running statistics only), so no label
    ever enters the assembled training inputs. The discriminative signal lives in the 256 features.
    """

    state = CountOnlineState.initial()
    rows: list[np.ndarray] = []
    for frame in range(features.shape[0]):
        rows.append(np.concatenate([features[frame], state.to_vector()]))
        state = state.update(features[frame], 0.0, False)
    return np.asarray(rows, dtype=np.float64)


def _causal_reestimates(
    gate: CountGate, features: np.ndarray, theta: float
) -> tuple[list[int], np.ndarray]:
    """Run the gate causally over a clip: return (reestimate_frames, p_trace) at threshold theta."""

    state = CountOnlineState.initial()
    reestimates: list[int] = []
    probs = np.empty(features.shape[0], dtype=np.float64)
    for frame in range(features.shape[0]):
        p = gate.infer(features[frame], state)
        probs[frame] = p
        did = p >= theta
        if did:
            reestimates.append(frame)
        state = state.update(features[frame], p, did)
    return reestimates, probs


def _train_count_gate(
    seed: int,
    train_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    config: RealCountBedConfig,
) -> tuple[CountGate, int]:
    """Train the one candidate gate on train-room value-of-computation targets. Returns gate, train frames."""

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(_assemble_inputs(features))
        targets.append(voc_targets_from_count_track(gt_by_clip[clip.clip_id], window=config.voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = CountGate(seed=seed)
    gate.fit(
        x, y, epochs=config.epochs, learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
    )
    return gate, int(x.shape[0])


# ---------------------------------------------------------------------------
# The real noisy-TV channel: white-noise audio featurized and marginal-matched to the real test content.
# ---------------------------------------------------------------------------


def _noise_seed(seed: int) -> int:
    payload = json.dumps(
        {"seed": int(seed), "key": "mop.beds.starss23.count.noisy_tv"},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b"mop-starss23-count-noisy-tv-v1\0" + payload).digest()[:4], "big")


def _matched_noise_features(
    noise_seed: int,
    n_frames: int,
    featurizer: Any,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Featurize deterministic white noise and match it to the test feature marginals."""

    rng = np.random.default_rng(noise_seed)
    audio = rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))
    features = featurizer.featurize(audio)
    mean = float(features.mean())
    std = float(features.std())
    if std > 0.0:
        features = (features - mean) / std * float(target_std) + float(target_mean)
    return features


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: FrozenCountFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Build the sealed count bed's deterministic aleatoric control channel."""

    return _matched_noise_features(_noise_seed(seed), n_frames, featurizer, target_mean, target_std)


# ---------------------------------------------------------------------------
# Per-seed run on the fixed real split.
# ---------------------------------------------------------------------------


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
    causal_reestimates: Callable[[Any, np.ndarray, float], tuple[list[int], np.ndarray]],
    score_rows: Callable[
        [list[tuple[str, list[int], list[int], list[int]]]], dict[str, Any]
    ],
) -> BudgetSeedRun:
    """Run the shared counting seed lifecycle with explicit gate and scoring providers."""

    gate, train_frames = train_gate()
    total_frames = int(sum(clip.n_frames for clip in test_clips))

    # A neutral-threshold causal pass over val gives the p distribution the budget grid is cut from.
    val_probs = np.concatenate(
        [causal_reestimates(gate, features_by_clip[clip.clip_id], 0.5)[1] for clip in val_clips]
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
            candidate_r, _ = causal_reestimates(gate, features, theta)
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

    # Preregistered operating point: the swept budget whose re-estimation rate is closest to the train-set
    # count-change density. A fixed rule set before scoring, using only train labels, never a val/test argmax.
    operating_budget_id = min(
        per_budget, key=lambda bid: abs(per_budget[bid]["rate"] - operating_density)
    )
    operating = per_budget[operating_budget_id]
    per_seed_block = {
        "seed": seed,
        "operating_budget_id": operating_budget_id,
        "clips": operating["clips"],
        "arm_scores": operating["arm_scores"],
    }

    operating_theta = operating["theta"]
    base_rate = operating["reestimations"][ARM_CANDIDATE] / max(1, total_frames)
    noise_reestimates, _ = causal_reestimates(gate, noise_features, operating_theta)
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
) -> BudgetSeedRun:
    """Bind the sealed gate and frame-micro referee to the shared counting seed lifecycle."""

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
        score_rows=_micro_count_score,
    )


# ---------------------------------------------------------------------------
# Harness arms across seeds.
# ---------------------------------------------------------------------------


def _flop_model(
    kind: str, total_frames: int, train_frames: int, config: RealCountBedConfig
) -> FlopModel:
    featurize = FLOPS_PER_FRAME_COUNT * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = FLOPS_PER_INFERENCE * total_frames if runs_gate else 0
    train = training_flops(train_frames, config.epochs) if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        train_flops=train,
    )


# ---------------------------------------------------------------------------
# Assemble and seal the real count artifact.
# ---------------------------------------------------------------------------


def build_real_count_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_PREREG_PATH,
) -> ArtifactResult:
    """Run the whole counting bed on the real STARSS23 subset and assemble the sealed artifact.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by
    the caller and never read from the wall clock inside a sealed body.
    """

    config = config or RealCountBedConfig()
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

    # Structural facts used by the SESOI cost-benefit and the operating-point rule. All label-only or
    # constant; no test score is read to build the prereg.
    train_count_clips = [count_clips[c.clip_id] for c in train_clips]
    test_count_clips = [count_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_count_clips)
    n_test_clips = len(test_clips)
    n_test_frames = int(sum(clip.n_frames for clip in test_clips))
    n_test_changes = int(sum(cc.n_changes for cc in test_count_clips))
    test_coast_from_zero = coast_from_zero_mae(test_count_clips)
    if n_test_changes == 0:
        raise CountProducerRefusal("the real test split carries no count changes to track")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_count_prereg(
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
    # Sign-flip statistic: delta_i = MAE_rate_matched_random(i) - MAE_candidate(i). Positive = candidate
    # reduces error. The exact_sign_flip test is one-sided upper tail on these deltas.
    deltas = [
        block["arm_scores"][PRIMARY_CONTROL]["mae"] - block["arm_scores"][ARM_CANDIDATE]["mae"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1=sesoi_mae, provisional=False)
    mean_delta_exceeds_sesoi = bool(sesoi.exceeds_sesoi)
    # Report-facing convention (task): candidate minus random, negative = candidate better (lower MAE).
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

    # Shared per-clip tracks, sealed once (identical across seeds): the ground-truth count track and the
    # frozen estimator track the verifier re-coasts every arm against.
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
            "question": (
                "concurrent-source counting (distinct from the seven sealed onset-localization nulls)"
            ),
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one real run is a mechanics demonstration; scientific confirmation needs the independent "
                "verifier plus at least three bias-independent reproductions and cannot be self-certified"
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
        "cold_start": COLD_START,
        "primary_control": PRIMARY_CONTROL,
        "corpus_tracks": corpus_tracks,
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
            "producer_schema": COUNT_PRODUCER_SCHEMA,
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


DEFAULT_COUNT_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_BED.json")
