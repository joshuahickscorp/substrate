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

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    mint_demonstration,
)
from mop.substrate.events import canonical_sha256

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter
from .artifact import (
    ARTIFACT_SCHEMA,
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
    STAGE3_REQUIREMENT_ID,
    BedConfig,
    _build_budget_points,
    _causal_fires,
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
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate import FLOPS_PER_INFERENCE, OnlineState
from .harness import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    run_matched_budget,
)
from .prereg import DEFAULT_PREREG_PATH, build_prereg, write_prereg
from .referee import score_arm
from .schema import COLLAR_FRAMES, N_CHANNELS, SAMPLES_PER_FRAME, Clip, ClipSplit
from .stats import exact_sign_flip

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


# ---------------------------------------------------------------------------
# The native fold-respecting split and one-time featurization.
# ---------------------------------------------------------------------------


def _fold_respecting_split(adapter: RealStarssAdapter, n_val_rooms: int) -> ClipSplit:
    """Build train / val / test respecting the native fold boundary: test is exactly fold-4 dev-test."""

    dev = adapter.dev_split()
    by_id = {clip.clip_id: clip for clip in adapter.clips()}
    fold3 = [by_id[cid] for cid in dev.dev_train]
    fold4 = [by_id[cid] for cid in dev.dev_test]
    fold3_rooms = sorted({clip.room_id for clip in fold3})
    if n_val_rooms <= 0 or n_val_rooms >= len(fold3_rooms):
        raise RealArtifactRefusal(
            f"n_val_rooms must leave at least one train room; saw {n_val_rooms} of {len(fold3_rooms)}"
        )
    val_rooms = set(fold3_rooms[-n_val_rooms:])
    train = tuple(sorted((c for c in fold3 if c.room_id not in val_rooms), key=lambda c: c.clip_id))
    val = tuple(sorted((c for c in fold3 if c.room_id in val_rooms), key=lambda c: c.clip_id))
    test = tuple(sorted(fold4, key=lambda c: c.clip_id))
    return ClipSplit(
        train=train,
        val=val,
        test=test,
        detail={
            "train_rooms": sorted({c.room_id for c in train}),
            "val_rooms": sorted(val_rooms),
            "test_rooms": sorted({c.room_id for c in test}),
            "split_rule": "test = native fold-4 dev-test; val = last N fold-3 rooms; train = rest of fold-3",
        },
    )


def _featurize_all(adapter: RealStarssAdapter, featurizer: FrozenFeaturizer) -> dict[str, np.ndarray]:
    """Featurize every real clip once with the frozen front-end. Reused across all paired seeds."""

    return {clip.clip_id: featurizer.featurize(adapter.audio(clip.clip_id)) for clip in adapter.clips()}


def _onset_density(clips: tuple[Clip, ...]) -> float:
    """Onset density (onsets per frame) over a set of clips. Label-only; reads no score."""

    onsets = sum(len(clip.onsets) for clip in clips)
    frames = sum(clip.n_frames for clip in clips)
    return onsets / frames if frames > 0 else 0.0


# ---------------------------------------------------------------------------
# The real noisy-TV channel: white-noise audio featurized and marginal-matched to the real test content.
# ---------------------------------------------------------------------------


def _noise_seed(seed: int) -> int:
    payload = json.dumps(
        {"seed": int(seed), "key": "mop.beds.starss23.real.noisy_tv"},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b"mop-starss23-real-noisy-tv-v1\0" + payload).digest()[:4], "big")


def _real_noisy_tv_features(
    seed: int,
    n_frames: int,
    featurizer: FrozenFeaturizer,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Pure-aleatoric channel: 4-channel white-noise audio featurized, then affine-matched to test marginals.

    White noise carries no reducible onset structure (no sharp signal-band flux attacks), so a gate that
    keys on the coherent onset signature fires at chance on it. Matching the channel's global feature mean
    and standard deviation to the real test content removes any raw-magnitude confound, so only a gate
    that chases irreducible novelty fires preferentially. Deterministic in the seed.
    """

    rng = np.random.default_rng(_noise_seed(seed))
    audio = rng.standard_normal((N_CHANNELS, n_frames * SAMPLES_PER_FRAME))
    features = featurizer.featurize(audio)
    mean = float(features.mean())
    std = float(features.std())
    if std > 0.0:
        features = (features - mean) / std * float(target_std) + float(target_mean)
    return features


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
        [_causal_fires(gate, features_by_clip[clip.clip_id], 0.5)[1] for clip in split.val]
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
            fires, _ = _causal_fires(gate, features_by_clip[clip.clip_id], theta)
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
            candidate_fires, _ = _causal_fires(gate, features, theta)
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
    noise_fires, _ = _causal_fires(gate, noise_features, operating_theta)
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


@dataclass(frozen=True, slots=True)
class RealBedArtifact:
    """The assembled sealed real-data bed artifact plus the mechanics-only demonstration receipt."""

    artifact: dict[str, Any]
    prereg: dict[str, Any]
    verdict: str
    detail: dict[str, Any]

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def build_real_bed_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_PREREG_PATH,
) -> RealBedArtifact:
    """Run the whole bed on the real STARSS23 subset and assemble the sealed artifact.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by
    the caller and never read from the wall clock inside a sealed body.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = FrozenFeaturizer()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    features_by_clip = _featurize_all(adapter, featurizer)
    split = _fold_respecting_split(adapter, config.n_val_rooms)

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
    prereg_written = write_prereg(prereg, prereg_path)
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

    budget_points = _build_budget_points(seed_runs, bed_config)
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
    stats_block = {
        "deltas": [float(value) for value in deltas],
        "t_obs": float(sign_flip.mean_delta),
        "one_sided_p": float(sign_flip.one_sided_p),
        "n_permutations": int(sign_flip.permutations),
        "two_sided_005_reachable": bool(sign_flip.two_sided_alpha_reachable),
        "sesoi_f1": sesoi_f1,
        "sesoi_provisional": False,
        "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
        "claim_verb": "consistent with",
        "experimental_unit": "clip",
        "frame_or_clip_bootstrap_allowed": False,
        "prereg_canonical_sha256": prereg["canonical_sha256"],
    }

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["firing_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls_block = {
        "noisy_tv_at_chance": noisy_tv_at_chance,
        "mean_firing_rate_on_noise": round(float(mean_noise_rate), 12),
        "mean_base_rate": round(float(mean_base_rate), 12),
        "per_seed_noisy_tv": [run.noisy_tv for run in seed_runs],
        "primary_control": PRIMARY_CONTROL,
        "control_arms": [ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv"],
    }
    flags_block = {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
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
    evidence_digest = canonical_sha256(core_evidence)
    receipt = mint_demonstration(
        mechanism_id=BED_ID,
        stage=STAGE,
        requirement_id=STAGE3_REQUIREMENT_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE, "noisy_tv"),
        evidence_digest=evidence_digest,
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

    body: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "collar_frames": COLLAR_FRAMES,
        "primary_control": PRIMARY_CONTROL,
        "source_kind": "real",
        "rights_clean": True,
        "reproductions": 0,
        "seeds": list(config.seeds),
        "per_seed": per_seed,
        "stats": stats_block,
        "controls": controls_block,
        "flags": flags_block,
        "verdict": verdict,
        "harness": report.payload(),
        "matched_budget": report.matched_budget.payload(),
        "matched_budget_wall_note": (
            "wall_ns is a deterministic nominal at a 1 GFLOP/s reference so the artifact is "
            "byte-reproducible; the measured wall is unsealed run provenance, and the authoritative "
            "sealed compute axes are the parameter count and the FLOP ledger"
        ),
        "break_even": report.break_even.payload(),
        "featurizer": {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        },
        "gate": {
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
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
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)

    return RealBedArtifact(
        artifact=body,
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
