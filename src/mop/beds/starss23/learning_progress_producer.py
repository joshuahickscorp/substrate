"""Real-data producer for the STARSS23 ESCS "learning_progress" gate variant.

This is a net-new, additive component. It runs the E1 learning_progress gate variant end to end on the
REAL, MIT-licensed STARSS23 FOA subset and assembles a byte-sealed
``proof/STARSS23_ESCS_BED_learning_progress.json`` in the same shape as the committed artifact. It
changes none of the sealed scoring logic: the referee, the matched-budget harness and its FLOP
accounting and Pareto analysis, the exact sign-flip statistics, and every control are imported unchanged
from their modules. Only the gate (the single trained module) is swapped for
``LearningProgressGate``, and the FLOP ledger is charged with that gate's own honest inference and
training cost.

Performance mandate: the deterministic featurizer is the dominant, gate-independent cost, so this
producer NEVER re-featurizes. It loads the shared feature cache (``load_cached_corpus``) built once over
the corpus and reuses it for every seed and every control. The FLOP ledger still charges the featurizer
per arm in full (caching is a wall-clock optimization, not a budget reduction).

Preregistration: the SESOI (0.05) and the sign-flip plan (five paired seeds, one-sided floor 1/32) are
NOT invented here; they are imported unchanged from the sealed ``prereg.py`` and
``gate_variants_prereg.py`` and re-stated in a self-sealed sidecar written BEFORE any test score is read.
The learning_progress hypothesis is an exploratory FIFTH variant outside the sealed four-variant family,
held to the identical bar; by the sealed promotion bar and the family-wise multiplicity control, a single
run, and any run at n equals 5, can never promote. Every artifact hardcodes activation_allowed=false,
scientific_promotion=false, and independent_scientific_confirmation=false.

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
from mop.science import ArtifactResult, demonstration_receipt, finalize_artifact, safety_flags
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    FlopModel,
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import (
    BOUNDED_CLAIM_VERB,
    FORBIDDEN_CLAIM_VERBS,
    exact_sign_flip,
    sign_flip_payload,
)
from mop.substrate.events import canonical_sha256, write_canonical_json

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .artifact import (
    ARTIFACT_SCHEMA,
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
)
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .experiments import ONSET_BUDGET_POLICY
from .feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate_learning_progress import (
    DEFAULT_EPOCHS,
    ONLINE_LR,
    LearningProgressGate,
)
from .prereg import (
    PREREG_DIRECTION,
    PREREG_METRIC,
    PREREGISTERED_SESOI_F1,
)
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealBedConfig,
    _real_noisy_tv_features,
)
from .referee import COLLAR_MS, COLLAR_RULE, MATCH_RULE, PR_RULE, score_arm
from .schema import COLLAR_FRAMES, Clip, ClipSplit

LP_PRODUCER_SCHEMA = "mop-starss23-escs-learning-progress-producer/v1"
VARIANT_ID = "learning_progress"

DEFAULT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_learning_progress.json")
DEFAULT_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_learning_progress.prereg.json")
DEFAULT_VARIANTS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_VARIANTS.prereg.json")


class LPProducerRefusal(ValueError):
    """Raised when the learning_progress producer cannot assemble a well-formed sealed artifact."""


# ---------------------------------------------------------------------------
# Preregistration sidecar. Reuses the sealed SESOI and sign-flip plan unchanged.
# ---------------------------------------------------------------------------


def _read_variants_prereg_digest(path: Path = DEFAULT_VARIANTS_PREREG_PATH) -> str | None:
    """Return the sealed E1 variant preregistration canonical digest, or None if absent."""

    import json

    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_bytes().decode("utf-8"))
    except (ValueError, OSError):
        return None
    digest = body.get("canonical_sha256")
    return str(digest) if isinstance(digest, str) else None


def build_lp_prereg(
    *,
    timestamp: str,
    n_test_clips: int,
    n_test_onsets: int,
    n_test_frames: int,
    train_onset_density: float,
    operating_firing_fraction: float,
    n_seeds: int,
) -> dict[str, Any]:
    """Assemble the self-sealed learning_progress prereg sidecar. Imports the sealed thresholds unchanged.

    Nothing here reads a test score. The SESOI and the sign-flip plan are the sealed values from
    ``prereg.py`` and are not modified; this sidecar only records that the exploratory learning_progress
    variant is held to the identical bar and can never promote from a single n equals 5 run.
    """

    permutations = 2**n_seeds
    body: dict[str, Any] = {
        "schema": "mop-starss23-escs-learning-progress-prereg/v1",
        "stage": STAGE,
        "bed_id": BED_ID,
        "claim_scope": CLAIM_SCOPE,
        "timestamp": timestamp,
        "preregistered_before_reading_test_scores": True,
        "wave": "E1 gate-variant iteration",
        "variant_id": VARIANT_ID,
        "variant_family_note": (
            "learning_progress is an exploratory fifth variant, outside the sealed four-variant family "
            "(refractory_nms, flux_novelty_target, energy_whitened_features, recency_gap_penalty). It is "
            "held to the identical SESOI and sign-flip plan, and the sealed promotion bar and family-wise "
            "multiplicity control apply at least as strictly: a single run, and any run at n equals 5, "
            "can never promote"
        ),
        "hypothesis": (
            "Fire on reducible surprise, not raw energy: an RND fixed-random-target predictor error whose "
            "derivative (learning progress) gates firing, so it stops firing on high-energy but unlearnable "
            "or steady regions (docs/ESCS_DEEP_RESEARCH.md lines 51 to 55)"
        ),
        "metric": PREREG_METRIC,
        "collar_frames": COLLAR_FRAMES,
        "collar_ms": COLLAR_MS,
        "direction": PREREG_DIRECTION,
        "primary_control": "rate_matched_random",
        "sesoi": {
            "sesoi_f1": round(float(PREREGISTERED_SESOI_F1), 12),
            "provisional": False,
            "source": "imported unchanged from sealed prereg.PREREGISTERED_SESOI_F1",
            "train_onset_density": round(float(train_onset_density), 12),
        },
        "operating_point_rule": (
            "the swept firing budget whose firing fraction is closest to the train-set onset density; a "
            "fixed rule set before scoring, using only train labels, never a val or test F1 argmax"
        ),
        "operating_firing_fraction": round(float(operating_firing_fraction), 12),
        "sign_flip_test_plan": {
            "test": "exact sign-flip permutation, one-sided, upper tail",
            "n_paired_seeds": n_seeds,
            "n_permutations": permutations,
            "statistic": "mean of paired per-seed F1 deltas (candidate minus rate_matched_random)",
            "min_one_sided_p": round(1.0 / permutations, 12),
            "two_sided_floor": round(2.0 / permutations, 12),
            "alpha": 0.05,
            "two_sided_alpha_reachable": (2.0 / permutations) <= 0.05,
            "phipson_smyth_applied": False,
        },
        "claim_ceiling": {
            "experimental_unit": "clip",
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": int(n_test_frames),
            "claim_verb": BOUNDED_CLAIM_VERB,
            "forbidden_verbs": list(FORBIDDEN_CLAIM_VERBS),
            "frame_or_clip_bootstrap_allowed": False,
        },
        "sealed_variants_prereg_canonical_sha256": _read_variants_prereg_digest(),
        "promotion_bar": (
            "promote only when the registered SESOI is exceeded AND the one-sided sign-flip p clears the "
            "Bonferroni-adjusted alpha AND at least three bias-independent reproductions triangulate the "
            "same direction; a single run, and any run at n equals 5, can never promote"
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    body["canonical_sha256"] = canonical_sha256(body)
    return body


# ---------------------------------------------------------------------------
# Training and per-seed run on the cached real corpus.
# ---------------------------------------------------------------------------


def _train_lp_gate(
    seed: int,
    split_train: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[LearningProgressGate, int]:
    """Fit the learning_progress gate on the pooled train-room features. Returns the gate and frame count."""

    train_features = np.concatenate(
        [features_by_clip[clip.clip_id] for clip in split_train], axis=0
    )
    gate = LearningProgressGate(seed=seed)
    gate.fit(train_features, epochs=epochs, learning_rate=learning_rate)
    return gate, int(train_features.shape[0])


def _onset_density(clips: tuple[Clip, ...]) -> float:
    onsets = sum(len(clip.onsets) for clip in clips)
    frames = sum(clip.n_frames for clip in clips)
    return onsets / frames if frames > 0 else 0.0


def _adjacency_fraction(fires: list[int]) -> tuple[int, int]:
    """Return (n_adjacent, n_fires): a fire is adjacent when another fire sits one frame away."""

    fire_set = set(fires)
    adjacent = sum(1 for f in fires if (f - 1) in fire_set or (f + 1) in fire_set)
    return adjacent, len(fires)


@dataclass(frozen=True, slots=True)
class _LPSeedRun:
    seed: int
    total_frames: int
    train_frames: int
    gate_params: int
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    per_seed_block: dict[str, Any]
    noisy_tv: dict[str, Any]
    diagnostics: dict[str, Any]


def _run_seed_lp(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: RealBedConfig,
    operating_density: float,
    *,
    epochs: int,
    learning_rate: float,
) -> _LPSeedRun:
    """Fit the gate for one seed, sweep the firing budget, score every arm on the fixed real test set."""

    gate, train_frames = _train_lp_gate(
        seed, split.train, features_by_clip, epochs=epochs, learning_rate=learning_rate
    )
    total_frames = int(sum(clip.n_frames for clip in split.test))

    val_probs = np.concatenate(
        [gate.causal_scores(features_by_clip[clip.clip_id]) for clip in split.val]
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
            fires, _ = gate.causal_fires(features_by_clip[clip.clip_id], theta)
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
        candidate_adjacent = 0
        candidate_fires_total = 0
        for clip in split.test:
            features = features_by_clip[clip.clip_id]
            gt = list(clip.onset_frames)
            candidate_fires, _ = gate.causal_fires(features, theta)
            adj, nfr = _adjacency_fraction(candidate_fires)
            candidate_adjacent += adj
            candidate_fires_total += nfr
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
        arm_scores = {
            kind: score_arm(pairs, COLLAR_FRAMES).payload() for kind, pairs in arm_clip_scores.items()
        }
        per_budget[budget_id] = {
            "theta": theta,
            "rate": rate,
            "val_f1": val_f1,
            "clips": clips_block,
            "arm_scores": arm_scores,
            "firings": firings,
            "candidate_adjacent": candidate_adjacent,
            "candidate_fires_total": candidate_fires_total,
        }

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
    noise_fires, _ = gate.causal_fires(noise_features, operating_theta)
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    adj = operating["candidate_adjacent"]
    nfr = operating["candidate_fires_total"]
    diagnostics = {
        "operating_budget_id": operating_budget_id,
        "candidate_fires": nfr,
        "candidate_adjacent_fires": adj,
        "candidate_adjacency_fraction": round(adj / nfr, 12) if nfr else 0.0,
        "candidate_distinct_onset_tp": operating["arm_scores"][ARM_CANDIDATE]["tp"],
        "rate_matched_random_distinct_onset_tp": operating["arm_scores"][ARM_RATE_MATCHED_RANDOM]["tp"],
    }

    return _LPSeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
        diagnostics=diagnostics,
    )

# ---------------------------------------------------------------------------
# FLOP ledger charged with the learning_progress gate's own honest cost.
# ---------------------------------------------------------------------------


def _flop_model_lp(
    kind: str,
    total_frames: int,
    train_frames: int,
    gate: LearningProgressGate,
    epochs: int,
) -> FlopModel:
    featurize = FLOPS_PER_FRAME * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = gate.flops_per_inference() * total_frames if runs_gate else 0
    train = gate.training_flops(train_frames, epochs) if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        train_flops=train,
    )


# ---------------------------------------------------------------------------
# Assemble and seal the learning_progress artifact.
# ---------------------------------------------------------------------------


def build_lp_bed_artifact(
    *,
    timestamp: str,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = ONLINE_LR,
    prereg_path: str | Path = DEFAULT_PREREG_PATH,
) -> ArtifactResult:
    """Run the learning_progress variant on the cached real corpus and assemble the sealed artifact."""

    config = config or RealBedConfig()
    corpus = load_cached_corpus(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=config.max_frames,
        n_val_rooms=config.n_val_rooms,
    )
    split = corpus.split
    features_by_clip = corpus.features_by_clip

    train_density = _onset_density(split.train)
    n_test_clips = len(split.test)
    n_test_onsets = sum(len(clip.onsets) for clip in split.test)
    n_test_frames = int(sum(clip.n_frames for clip in split.test))
    if n_test_onsets == 0:
        raise LPProducerRefusal("the real test split carries no onsets to score")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the sidecar (reusing the sealed SESOI and sign-flip plan) BEFORE any test score.
    prereg = build_lp_prereg(
        timestamp=timestamp,
        n_test_clips=n_test_clips,
        n_test_onsets=n_test_onsets,
        n_test_frames=n_test_frames,
        train_onset_density=train_density,
        operating_firing_fraction=operating_rate,
        n_seeds=len(config.seeds),
    )
    prereg_written = write_canonical_json(prereg, prereg_path)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])

    # 2. Now run the paired seeds and score the test split. The noisy-TV channel reuses the sealed real
    # producer helper (white-noise audio featurized and affine-matched to the test marginals).
    featurizer = FrozenFeaturizer()
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in split.test], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_LPSeedRun] = []
    reference_gate: LearningProgressGate | None = None
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        run = _run_seed_lp(
            seed,
            split,
            features_by_clip,
            noise_features,
            config,
            train_density,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        seed_runs.append(run)
        if reference_gate is None:
            reference_gate = LearningProgressGate(seed=seed)
    measured_wall_ns = max(1, time.perf_counter_ns() - started)
    assert reference_gate is not None

    budget_points = build_budget_points(
        ONSET_BUDGET_POLICY, seed_runs, score_group="arm_scores", score_field="f1",
        action_group="firings",
        flop_model=lambda kind: _flop_model_lp(
            kind, seed_runs[0].total_frames, seed_runs[0].train_frames, reference_gate, epochs
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
    beats_random = bool(sign_flip.one_sided_significant and mean_delta_exceeds_sesoi)
    stats_block = sign_flip_payload(
        sign_flip, deltas, sesoi_key="sesoi_f1", sesoi=sesoi_f1,
        exceeds_sesoi=mean_delta_exceeds_sesoi, provisional=False,
        prereg_digest=prereg["canonical_sha256"],
        extra={"beats_rate_matched_random": beats_random},
    )

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["firing_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls_block = noise_control_summary(
        ONSET_BUDGET_POLICY, seed_runs, at_chance=noisy_tv_at_chance, mean_noise_rate=mean_noise_rate,
        mean_base_rate=mean_base_rate, rate_key="mean_firing_rate_on_noise",
    )

    # Fire-spread diagnostics pooled across seeds at the operating budget.
    total_candidate_fires = sum(run.diagnostics["candidate_fires"] for run in seed_runs)
    total_candidate_adjacent = sum(run.diagnostics["candidate_adjacent_fires"] for run in seed_runs)
    total_candidate_tp = sum(run.diagnostics["candidate_distinct_onset_tp"] for run in seed_runs)
    total_rmr_tp = sum(run.diagnostics["rate_matched_random_distinct_onset_tp"] for run in seed_runs)
    diagnostics_block = {
        "operating_point_rule": "swept budget closest to the train onset density",
        "pooled_over_seeds": {
            "candidate_fires": total_candidate_fires,
            "candidate_adjacent_fires": total_candidate_adjacent,
            "candidate_adjacency_fraction": (
                round(total_candidate_adjacent / total_candidate_fires, 12)
                if total_candidate_fires
                else 0.0
            ),
            "candidate_distinct_onset_tp": total_candidate_tp,
            "rate_matched_random_distinct_onset_tp": total_rmr_tp,
            "committed_baseline_distinct_onset_tp": 204,
            "committed_random_distinct_onset_tp": 237,
        },
        "per_seed": [run.diagnostics for run in seed_runs],
    }

    flags_block = safety_flags()

    dominates = report.candidate_strictly_dominates_rate_matched_random
    meets_bar = dominates and sign_flip.one_sided_significant and mean_delta_exceeds_sesoi
    verdict = VERDICT_MECHANICS_OK if meets_bar else VERDICT_NULL

    referee_block = {
        "collar_frames": COLLAR_FRAMES,
        "collar_ms": COLLAR_MS,
        "collar_rule": COLLAR_RULE,
        "match_rule": MATCH_RULE,
        "pr_rule": PR_RULE,
    }

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
            "variant_id": VARIANT_ID,
            "forcing_null": STAGE3_FORCING_NULL,
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one real run of an exploratory gate variant is a mechanics demonstration; scientific "
                "confirmation needs the independent verifier plus at least three bias-independent "
                "reproductions and cannot be self-certified"
            ),
        },
    )

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
        "variant": {
            "variant_id": VARIANT_ID,
            "producer_schema": LP_PRODUCER_SCHEMA,
            "exploratory_fifth_variant": True,
            "hypothesis": prereg["hypothesis"],
        },
        "seeds": list(config.seeds),
        "per_seed": per_seed,
        "stats": stats_block,
        "controls": controls_block,
        "fire_spread_diagnostic": diagnostics_block,
        "referee": referee_block,
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
            "variant_id": VARIANT_ID,
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": reference_gate.state_bytes(),
            "flops_per_inference": reference_gate.flops_per_inference(),
            "parameter_digest": reference_gate.parameter_digest(),
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_firing": DOWNSTREAM_FLOPS_PER_FIRING,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // DOWNSTREAM_FLOPS_PER_FIRING,
        },
        "real_corpus": {
            "producer_schema": LP_PRODUCER_SCHEMA,
            "cache_key": corpus.cache_key,
            "cache_dir": str(corpus.cache_dir),
            "featurizer_digest": corpus.featurizer_digest,
            "foa_root": str(Path(foa_root)),
            "metadata_root": str(Path(metadata_root)),
            "n_clips": len(corpus.clips),
            "split_rooms": dict(split.detail),
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": n_test_frames,
            "train_onset_density": round(float(train_density), 12),
            "operating_firing_fraction": round(float(operating_rate), 12),
        },
        "prereg": {
            "path": str(prereg_written),
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_f1": sesoi_f1,
            "provisional": False,
            "written_before_test_scores": True,
            "sealed_variants_prereg_canonical_sha256": prereg[
                "sealed_variants_prereg_canonical_sha256"
            ],
        },
        "demonstration_receipt": receipt,
    }
    return finalize_artifact(
        body,
        prereg=prereg,
        verdict=verdict,
        detail={
            "beats_random": beats_random,
            "dominates": dominates,
            "mean_delta": float(sign_flip.mean_delta),
            "one_sided_p": float(sign_flip.one_sided_p),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_f1": sesoi_f1,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(v) for v in deltas],
            "pooled_adjacency_fraction": diagnostics_block["pooled_over_seeds"][
                "candidate_adjacency_fraction"
            ],
            "pooled_candidate_distinct_onset_tp": total_candidate_tp,
            "pooled_rate_matched_random_distinct_onset_tp": total_rmr_tp,
        },
    )
