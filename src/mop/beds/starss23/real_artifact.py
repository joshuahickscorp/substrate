"""Real-data producer for the STARSS23 ESCS event-formation bed.

This is a net-new, additive component. It runs the whole bed end to end on the REAL, MIT-licensed
STARSS23 FOA subset served by ``RealStarssAdapter`` and assembles the same byte-sealed
``proof/STARSS23_ESCS_BED.json`` schema the independent verifier re-scores from specification. It changes
none of the sealed scoring logic: the referee, the harness FLOP accounting and Pareto analysis, the exact
sign-flip statistics, and every control are imported unchanged from their modules, and the value-of-
computation training-target assembly, the causal firing pass, the FLOP model, and the budget-point
assembly are imported unchanged from the synthetic producer so the real lane scores byte-identically to
the tested synthetic lane.

Two things differ from the synthetic producer, both because the corpus is now fixed real data rather than
a per-seed regenerated fixture:

1. The adapter is built once and every clip is featurized once; only the trained gate and the random
   controls vary across the five paired seeds. Data, split, and ground truth are identical across seeds.
2. The room-disjoint split respects the native STARSS23 fold boundary: the score partition is exactly the
   fold-4 dev-test rooms, and the val rooms are carved from the fold-3 dev-train rooms, so train, val, and
   test are room-disjoint and clip-disjoint and the test set is the held-out native dev-test.

The SESOI is preregistered before any test score is read (see ``prereg.py``); this producer writes the
sealed prereg first and records its digest in the artifact. The verdict is a mechanics demonstration
only: ``activation_allowed``, ``scientific_promotion``, and ``independent_scientific_confirmation`` are
hardcoded false, and a single run can never be scientifically confirmed (that needs the independent
verifier plus at least three bias-independent reproductions).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
import time
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
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.gating import causal_gate_trace
from mop.science.statistics import exact_sign_flip, sign_flip_payload
from mop.substrate.events import write_canonical_json

from . import BED_ID, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, domain_seed, map_clip_audio, marginal_matched_noise, native_fold_split
from .artifact import (
    ARTIFACT_SCHEMA,
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    BedConfig,
    _flop_model,
    _pooled_score,
    _SeedRun,
    _train_gate,
)
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .experiments import ONSET_BUDGET_POLICY
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate import FLOPS_PER_INFERENCE, OnlineState
from .prereg import DEFAULT_PREREG_PATH, build_prereg
from .referee import score_arm
from .schema import COLLAR_FRAMES, Clip, ClipSplit

REAL_PRODUCER_SCHEMA = "mop-starss23-escs-real-producer/v1"

# The real STARSS23 FOA subset and metadata roots on this host.
DEFAULT_FOA_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/foa_subset/foa_dev")
DEFAULT_METADATA_ROOT = Path(
    "/Users/scammermike/Downloads/mop-data/starss23/metadata_dev_extracted/metadata_dev"
)

# Val rooms are carved from the fold-3 dev-train rooms so train, val, and test stay room-disjoint and the
# test partition is exactly the held-out fold-4 dev-test.
DEFAULT_N_VAL_ROOMS = 2


class RealArtifactRefusal(ValueError):
    """Raised when the real producer cannot assemble a well-formed sealed artifact."""


@dataclass(frozen=True, slots=True)
class RealBedConfig:
    """Real-run configuration. The paired seeds and the sweep mirror the recipe; the data is fixed."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS
    target_rates: tuple[float, ...] = (0.10, 0.08, 0.06)
    noisy_tv_frames: int = 2000
    max_frames: int | None = None

    def bed_config(self) -> BedConfig:
        """A BedConfig carrying only the fields the reused synthetic helpers read (epochs, lr, etc.)."""

        return BedConfig(
            seeds=self.seeds,
            target_rates=self.target_rates,
            noisy_tv_frames=self.noisy_tv_frames,
            downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        )


def _onset_density(clips: tuple[Clip, ...]) -> float:
    """Onset density (onsets per frame) over a set of clips. Label-only; reads no score."""

    onsets = sum(len(clip.onsets) for clip in clips)
    frames = sum(clip.n_frames for clip in clips)
    return onsets / frames if frames > 0 else 0.0


# ---------------------------------------------------------------------------
# The real noisy-TV channel: white-noise audio featurized and marginal-matched to the real test content.
# ---------------------------------------------------------------------------


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: FrozenFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Build the real onset bed's independently seeded aleatoric control channel."""

    noise_seed = domain_seed(
        seed, "mop.beds.starss23.real.noisy_tv", b"mop-starss23-real-noisy-tv-v1"
    )
    return marginal_matched_noise(noise_seed, n_frames, featurizer, target_mean, target_std)


# ---------------------------------------------------------------------------
# Per-seed run on the fixed real split.
# ---------------------------------------------------------------------------


def _run_seed_real(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: BedConfig,
    operating_density: float,
) -> _SeedRun:
    """Train the gate for one seed, sweep the firing budget, score every arm on the fixed real test set."""

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
            candidate_fires, _ = causal_gate_trace(
                gate, features, theta, OnlineState.initial
            )
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

    # Preregistered operating point: the swept firing budget whose rate is closest to the train-set onset
    # density. A fixed rule set before scoring, using only train labels, never a val or test F1 argmax.
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
    base_rate = operating["firings"][ARM_CANDIDATE] / max(1, total_frames)
    noise_fires, _ = causal_gate_trace(
        gate, noise_features, operating_theta, OnlineState.initial
    )
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
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


# ---------------------------------------------------------------------------
# Assemble and seal the real artifact.
# ---------------------------------------------------------------------------


def build_real_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_PREREG_PATH,
) -> ArtifactResult:
    """Run the whole bed on the real STARSS23 subset and assemble the sealed artifact.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by
    the caller and never read from the wall clock inside a sealed body.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = FrozenFeaturizer()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    split = native_fold_split(
        adapter, config.n_val_rooms, refusal=RealArtifactRefusal, refuse_empty=False
    )

    # Structural facts used by the SESOI cost-benefit and the operating-point rule. All label-only or
    # constant; no test score is read to build the prereg.
    train_density = _onset_density(split.train)
    n_test_clips = len(split.test)
    n_test_onsets = sum(len(clip.onsets) for clip in split.test)
    n_test_frames = int(sum(clip.n_frames for clip in split.test))
    if n_test_onsets == 0:
        raise RealArtifactRefusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_prereg(
        timestamp=timestamp,
        operating_firing_fraction=operating_rate,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
        train_onset_density=train_density,
        n_test_frames=n_test_frames,
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])

    # 2. Now run the paired seeds and score the test split.
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in split.test], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_SeedRun] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_runs.append(
            _run_seed_real(seed, split, features_by_clip, noise_features, bed_config, train_density)
        )
    measured_wall_ns = max(1, time.perf_counter_ns() - started)

    budget_points = build_budget_points(
        ONSET_BUDGET_POLICY, seed_runs, score_group="arm_scores", score_field="f1",
        action_group="firings",
        flop_model=lambda kind: _flop_model(
            kind, seed_runs[0].total_frames, seed_runs[0].train_frames, bed_config
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
    deltas = [
        block["arm_scores"][ARM_CANDIDATE]["f1"] - block["arm_scores"][PRIMARY_CONTROL]["f1"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    mean_delta_exceeds_sesoi = bool(sign_flip.mean_delta >= sesoi_f1)
    stats_block = sign_flip_payload(
        sign_flip, deltas, sesoi_key="sesoi_f1", sesoi=sesoi_f1,
        exceeds_sesoi=mean_delta_exceeds_sesoi, provisional=False,
        prereg_digest=prereg["canonical_sha256"],
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
        mechanism_id=BED_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv"),
        evidence=core_evidence,
        verdict=verdict,
        detail={
            "source_kind": "real",
            "forcing_null": STAGE3_FORCING_NULL,
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
            "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing,
        },
        "real_corpus": {
            "producer_schema": REAL_PRODUCER_SCHEMA,
            "foa_root": str(Path(foa_root)),
            "metadata_root": str(Path(metadata_root)),
            "n_clips": len(adapter.clips()),
            "split_rooms": split.detail,
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": n_test_frames,
            "train_onset_density": round(float(train_density), 12),
            "operating_firing_fraction": round(float(operating_rate), 12),
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
            "sesoi_f1": sesoi_f1,
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
            "mean_delta": float(sign_flip.mean_delta),
            "one_sided_p": float(sign_flip.one_sided_p),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_f1": sesoi_f1,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(v) for v in deltas],
        },
    )
