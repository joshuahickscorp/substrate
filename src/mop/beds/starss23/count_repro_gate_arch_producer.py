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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    mint_demonstration,
)
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter
from .count_controls import (
    always_on_fires,
    at_chance,
    never_update_reestimates,
    rate_matched_random_fires,
)
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_harness import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    COUNT_BED_ID,
    CountArm,
    CountArmSeedResult,
    CountBudgetPoint,
    FlopModel,
    run_matched_budget,
)
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_gate import CountOnlineState
# Gate-agnostic pipeline stages reused by reference from the sealed producer so they cannot drift.
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    DEFAULT_N_VAL_ROOMS,
    RealCountBedConfig,
    _assemble_inputs,
    _estimate_all,
    _featurize_all,
    _fold_respecting_split,
    _real_noisy_tv_features,
)
from .count_referee import COLD_START, score_arm
from .count_repro_gate_arch_gate import (
    FLOPS_PER_INFERENCE_GATE_ARCH,
    HIDDEN1,
    HIDDEN2,
    N_OUT,
    REPRO_AXIS,
    CountReproGateArchGate,
    D_IN_GATE_ARCH,
    training_flops_two_layer,
    voc_targets_from_count_track,
)
from .count_repro_gate_arch_prereg import (
    COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
    DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
    build_count_repro_gate_arch_prereg,
    write_count_repro_gate_arch_prereg,
)
from .schema import Clip
from .stats import BOUNDED_CLAIM_VERB, exact_sign_flip, sesoi_check

COUNT_REPRO_GATE_ARCH_PRODUCER_SCHEMA = "mop-starss23-count-repro-gate-arch-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-gate-arch/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The disjoint gate-architecture seed family: no overlap with the original (0..4), so no shared seed luck.
GATE_ARCH_SEEDS: tuple[int, ...] = (40, 41, 42, 43, 44)


class CountReproGateArchProducerRefusal(ValueError):
    """Raised when the gate-architecture reproduction cannot assemble a well-formed sealed artifact."""


# ---------------------------------------------------------------------------
# Gate-touching passes: re-authored around the two-layer gate. Bodies mirror the sealed producer exactly
# except that they construct and charge the re-authored architecture.
# ---------------------------------------------------------------------------


def _causal_reestimates(
    gate: CountReproGateArchGate, features: np.ndarray, theta: float
) -> tuple[list[int], np.ndarray]:
    """Run the gate causally over a clip: return (reestimate_frames, p_trace) at threshold theta.

    Identical control flow to the sealed producer's causal pass; only the gate object differs. The online
    state is the sealed ``CountOnlineState``, advanced from the gate's own decisions and never a label.
    """

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
) -> tuple[CountReproGateArchGate, int]:
    """Train the re-authored two-layer gate on train-room value-of-computation targets.

    The input assembly (``_assemble_inputs``) and the target rule (``voc_targets_from_count_track``) are the
    sealed, held-fixed pieces; only the gate class is the varied axis.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(_assemble_inputs(features))
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


@dataclass(frozen=True, slots=True)
class _ReproSeedRun:
    seed: int
    total_frames: int
    train_frames: int
    gate_params: int
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    per_seed_block: dict[str, Any]
    noisy_tv: dict[str, Any]


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
) -> _ReproSeedRun:
    """Train the gate for one seed, sweep the budget, score every arm on the fixed real test set."""

    gate, train_frames = _train_count_gate(seed, train_clips, features_by_clip, gt_by_clip, config)
    total_frames = int(sum(clip.n_frames for clip in test_clips))

    val_probs = np.concatenate(
        [_causal_reestimates(gate, features_by_clip[clip.clip_id], 0.5)[1] for clip in val_clips]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    for rate in config.target_rates:
        theta = float(np.quantile(val_probs, 1.0 - rate))
        budget_id = f"rate_{rate:.2f}"

        arm_clip_scores: dict[str, list[tuple[list[int], list[int], list[int]]]] = {
            ARM_CANDIDATE: [],
            ARM_RATE_MATCHED_RANDOM: [],
            ARM_ALWAYS_ON: [],
            ARM_NEVER_UPDATE: [],
        }
        reestimations = {kind: 0 for kind in arm_clip_scores}
        clips_block: list[dict[str, Any]] = []
        for clip in test_clips:
            features = features_by_clip[clip.clip_id]
            gt = list(gt_by_clip[clip.clip_id])
            estimator = [int(v) for v in estimator_by_clip[clip.clip_id].tolist()]
            candidate_r, _ = _causal_reestimates(gate, features, theta)
            arm_r = {
                ARM_CANDIDATE: candidate_r,
                ARM_RATE_MATCHED_RANDOM: rate_matched_random_fires(
                    candidate_r, clip.n_frames, seed=seed, clip_id=clip.clip_id
                ),
                ARM_ALWAYS_ON: always_on_fires(clip.n_frames),
                ARM_NEVER_UPDATE: never_update_reestimates(clip.n_frames),
            }
            for kind, r in arm_r.items():
                arm_clip_scores[kind].append((gt, estimator, list(r)))
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
            kind: score_arm(pairs, COLD_START).payload() for kind, pairs in arm_clip_scores.items()
        }
        per_budget[budget_id] = {
            "theta": theta,
            "rate": rate,
            "clips": clips_block,
            "arm_scores": arm_scores,
            "reestimations": reestimations,
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
    base_rate = operating["reestimations"][ARM_CANDIDATE] / max(1, total_frames)
    noise_reestimates, _ = _causal_reestimates(gate, noise_features, operating_theta)
    noise_rate = len(noise_reestimates) / noise_features.shape[0]
    noisy_tv = {
        "reestimate_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    return _ReproSeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
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


def _build_budget_points(
    seed_runs: list[_ReproSeedRun], config: RealCountBedConfig
) -> list[CountBudgetPoint]:
    total_frames = seed_runs[0].total_frames
    train_frames = seed_runs[0].train_frames
    gate_params = seed_runs[0].gate_params
    budget_points: list[CountBudgetPoint] = []
    for budget_id in seed_runs[0].per_budget:
        arms: dict[str, CountArm] = {}
        for kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE):
            seed_results = tuple(
                CountArmSeedResult(
                    seed=run.seed,
                    mae=run.per_budget[budget_id]["arm_scores"][kind]["mae"],
                    reestimations=run.per_budget[budget_id]["reestimations"][kind],
                )
                for run in seed_runs
            )
            arms[kind] = CountArm(
                name=f"{kind}@{budget_id}",
                kind=kind,
                total_frames=total_frames,
                params=gate_params if kind == ARM_CANDIDATE else 0,
                flop_model=_flop_model(kind, total_frames, train_frames, config),
                seed_results=seed_results,
            )
        budget_points.append(
            CountBudgetPoint(
                budget_id=budget_id,
                candidate=arms[ARM_CANDIDATE],
                rate_matched_random=arms[ARM_RATE_MATCHED_RANDOM],
                always_on=arms[ARM_ALWAYS_ON],
                never_update=arms[ARM_NEVER_UPDATE],
            )
        )
    return budget_points


# ---------------------------------------------------------------------------
# Assemble and seal the gate-architecture reproduction artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealCountReproGateArchArtifact:
    """The assembled sealed gate-architecture reproduction artifact plus the mechanics-only receipt."""

    artifact: dict[str, Any]
    prereg: dict[str, Any]
    verdict: str
    detail: dict[str, Any]

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def build_real_count_repro_gate_arch_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
) -> RealCountReproGateArchArtifact:
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

    features_by_clip = _featurize_all(adapter, featurizer)
    estimator_by_clip = _estimate_all(adapter, estimator)
    train_clips, val_clips, test_clips, split_detail = _fold_respecting_split(adapter, config.n_val_rooms)

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
    prereg_written = write_count_repro_gate_arch_prereg(prereg, prereg_path)
    sesoi_mae = float(prereg["sesoi"]["sesoi_mae"])

    # 2. Now run the paired seeds and score the test split.
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in test_clips], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_ReproSeedRun] = []
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

    budget_points = _build_budget_points(seed_runs, config)
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

    stats_block = {
        "metric": "coasted-count-MAE",
        "delta_definition": "delta_i = MAE_rate_matched_random(i) - MAE_candidate(i); positive = candidate lower error",
        "deltas": [float(value) for value in deltas],
        "t_obs": float(sign_flip.mean_delta),
        "mean_delta_control_minus_candidate": float(sign_flip.mean_delta),
        "mean_delta_candidate_minus_control": mean_delta_candidate_minus_random,
        "one_sided_p": float(sign_flip.one_sided_p),
        "n_permutations": int(sign_flip.permutations),
        "two_sided_005_reachable": bool(sign_flip.two_sided_alpha_reachable),
        "sesoi_mae": sesoi_mae,
        "sesoi_provisional": False,
        "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
        "claim_verb": BOUNDED_CLAIM_VERB,
        "experimental_unit": "clip",
        "frame_or_clip_bootstrap_allowed": False,
        "prereg_canonical_sha256": prereg["canonical_sha256"],
    }

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["reestimate_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    noisy_tv_at_chance = at_chance(min(1.0, mean_noise_rate), min(1.0, mean_base_rate))
    controls_block = {
        "noisy_tv_at_chance": noisy_tv_at_chance,
        "mean_reestimate_rate_on_noise": round(float(mean_noise_rate), 12),
        "mean_base_rate": round(float(mean_base_rate), 12),
        "per_seed_noisy_tv": [run.noisy_tv for run in seed_runs],
        "primary_control": PRIMARY_CONTROL,
        "control_arms": [ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"],
    }
    flags_block = {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }

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
    evidence_digest = canonical_sha256(core_evidence)
    receipt = mint_demonstration(
        mechanism_id=COUNT_BED_ID,
        stage=STAGE,
        requirement_id=STAGE3_REQUIREMENT_ID,
        controls_cleared=(ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"),
        evidence_digest=evidence_digest,
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

    body: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "stage": STAGE,
        "bed_id": COUNT_BED_ID,
        "reproduction_axis": REPRO_AXIS,
        "claim_scope": CLAIM_SCOPE,
        "cold_start": COLD_START,
        "primary_control": PRIMARY_CONTROL,
        "source_kind": "real",
        "rights_clean": True,
        "reproductions": 0,
        "seeds": list(config.seeds),
        "corpus_tracks": corpus_tracks,
        "per_seed": per_seed,
        "stats": stats_block,
        "controls": controls_block,
        "flags": flags_block,
        "verdict": verdict,
        "survives_operating_point": survives,
        "operating_point": {
            "budget_id": operating_budget_id,
            "candidate_mean_mae": candidate_op_mae,
            "rate_matched_random_mean_mae": rmr_op_mae,
            "candidate_strictly_beats_rate_matched_random": operating_candidate_beats_rmr,
        },
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
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
        },
        "estimator": {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        },
        "gate": {
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
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)

    return RealCountReproGateArchArtifact(
        artifact=body,
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


def write_count_repro_gate_arch_artifact(
    artifact: dict[str, Any],
    out_path: str | Path = DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH,
) -> Path:
    """Write the sealed artifact as canonical JSON bytes so its on-disk digest is reproducible."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact))
    return path
