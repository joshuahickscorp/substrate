"""Real-data producer for the featurizer_estimator bias-independent reproduction of the counting bed.

This is a NET-NEW, ADDITIVE component. It edits no sealed count_* or onset module and no existing proof. It
runs the whole counting bed end to end on the REAL, MIT-licensed STARSS23 FOA subset, but with the frozen
front-end swapped for the re-authored gammatone-plus-relative-flux featurizer and the frozen estimator
swapped for the re-authored cumulative-energy count estimator, holding everything else identical to the
sealed bed: the room-fold split, the trained count gate architecture and objective, the coasted-count-MAE
referee, the four controls, the exact sign-flip, and the matched-budget FLOP accounting are all imported
unchanged. Its disjoint seed family is (20, 21, 22, 23, 24), so it shares none of the sealed run's seed luck.

It assembles the byte-sealed ``proof/STARSS23_COUNTING_REPRO_featurizer_estimator.json`` a separately
authored, standard-library-only verifier re-scores from specification. The reproduction's SESOI is
preregistered before any test score is read (see ``count_repro_featurizer_estimator_prereg.py``); this
producer writes the sealed prereg first and records its digest in the artifact. The verdict is a mechanics
demonstration only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, ``reproductions`` is 0, and a single run can
never be scientifically confirmed no matter how clean the arithmetic.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

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
    BudgetSeedRun,
    FlopModel,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import count_sign_flip_payload, exact_sign_flip, sesoi_check
from mop.substrate.events import write_canonical_json

from . import FLOP_CEILING, STAGE3_FORCING_NULL
from .controls import (
    at_chance,
)
from .count_gate import (
    COUNT_VOC_WINDOW,
    FLOPS_PER_INFERENCE,
    CountOnlineState,
)
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    CountProducerRefusal,
    _causal_reestimates,
    _matched_noise_features,
    _micro_count_score,
    _train_count_gate,
    run_count_seed,
)
from .count_referee import COLD_START
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
from .experiments import COUNT_BED_ID, COUNT_BUDGET_POLICY
from .gate import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PONDER_LAMBDA, training_flops
from .schema import Clip

COUNT_REPRO_FE_PRODUCER_SCHEMA = "mop-starss23-count-repro-featurizer-estimator-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-featurizer-estimator/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

DEFAULT_N_VAL_ROOMS = 2


class CountReproProducerRefusal(ValueError):
    """Raised when the reproduction producer cannot assemble a well-formed sealed artifact."""


@dataclass(frozen=True, slots=True)
class ReproCountBedConfig:
    """Reproduction-run configuration. The disjoint seed family and the sweep mirror the sealed recipe."""

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


# ---------------------------------------------------------------------------
# The real noisy-TV channel, re-featurized with the re-authored front-end.
# ---------------------------------------------------------------------------


def _noise_seed(seed: int) -> int:
    payload = json.dumps(
        {"seed": int(seed), "key": "mop.beds.starss23.count_repro_featurizer_estimator.noisy_tv"},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(b"mop-starss23-count-repro-featurizer-estimator-noisy-tv-v1\0" + payload).digest()[:4],
        "big",
    )


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: ReproCountFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Build the swapped front-end's independently seeded aleatoric control channel."""

    return _matched_noise_features(_noise_seed(seed), n_frames, featurizer, target_mean, target_std)


# ---------------------------------------------------------------------------
# Per-seed run on the fixed real split.
# ---------------------------------------------------------------------------


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
    """Bind the swapped providers to the held-fixed counting seed lifecycle."""

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


def _flop_model(kind: str, total_frames: int, train_frames: int, config: ReproCountBedConfig) -> FlopModel:
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
# Assemble and seal the reproduction artifact.
# ---------------------------------------------------------------------------


def build_repro_count_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: ReproCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_REPRO_PREREG_PATH,
) -> ArtifactResult:
    """Run the whole counting bed reproduction on the real STARSS23 subset and assemble the sealed artifact.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by the
    caller and never read from the wall clock inside a sealed body.
    """

    from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split

    config = config or ReproCountBedConfig()
    featurizer = ReproCountFeaturizer()
    estimator = ReproCountEstimator()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    count_clips = build_count_clips(adapter, metadata_root)
    gt_by_clip = {cid: cc.count_track for cid, cc in count_clips.items()}

    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    estimator_by_clip = map_clip_audio(adapter, estimator.estimate_track)
    split = native_fold_split(adapter, config.n_val_rooms, refusal=CountProducerRefusal)
    train_clips, val_clips, test_clips = split.train, split.val, split.test
    split_detail = dict(split.detail)

    # Structural facts used by the SESOI cost-benefit and the operating-point rule. Label-only or constant.
    train_count_clips = [count_clips[c.clip_id] for c in train_clips]
    test_count_clips = [count_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_count_clips)
    n_test_clips = len(test_clips)
    n_test_frames = int(sum(clip.n_frames for clip in test_clips))
    n_test_changes = int(sum(cc.n_changes for cc in test_count_clips))
    test_coast_from_zero = coast_from_zero_mae(test_count_clips)
    if n_test_changes == 0:
        raise CountReproProducerRefusal("the real test split carries no count changes to track")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_repro_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        n_test_frames=n_test_frames,
        train_change_density=train_density,
        coast_from_zero_mae=test_coast_from_zero,
        c_reest_flops=config.downstream_flops_per_reestimate,
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
            "reproduction_axis": REPRO_AXIS,
            "forcing_null": STAGE3_FORCING_NULL,
            "question": "concurrent-source counting, featurizer_estimator bias-independent reproduction",
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
            "schema": COUNT_REPRO_FE_FEATURIZER_SCHEMA,
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
            "note": "re-authored gammatone ERB filterbank plus causal relative spectral flux",
        }, gate={
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
            "note": "held fixed, identical to the sealed count gate",
        }, receipt_payload=receipt,
        extra={
        "reproduction_axis": REPRO_AXIS,
        "reproduces": "proof/STARSS23_COUNTING_BED.json",
        "cold_start": COLD_START,
        "primary_control": PRIMARY_CONTROL,
        "corpus_tracks": corpus_tracks,
        "estimator": {
            "schema": COUNT_REPRO_FE_ESTIMATOR_SCHEMA,
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
            "note": "re-authored cumulative-energy (proportion-of-variance) count estimator, BETA=0.90",
        },
        "real_corpus": {
            "producer_schema": COUNT_REPRO_FE_PRODUCER_SCHEMA,
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


DEFAULT_REPRO_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_featurizer_estimator.json")
