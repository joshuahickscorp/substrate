#!/usr/bin/env python3
"""E1 variant producer: run the diversity_reg gate on the cached real STARSS23 corpus and seal the proof.

This is a net-new, additive entrypoint. It changes no sealed scoring logic and edits no committed file. It
runs the ``diversity_reg`` gate variant (src/mop/beds/starss23/gate_diversity_reg.py) through the EXISTING
sealed harness, referee, controls, and statistics on the shared feature cache, and writes the sealed
``proof/STARSS23_ESCS_BED_diversity_reg.json`` with the same schema as the committed real artifact.

What is reused unchanged (imported, never reimplemented):
  * the frozen feature cache (load_cached_corpus): the featurizer is computed ONCE over the corpus and
    every arm reads the cache, so this variant run is featurize-free. The FLOP ledger still charges the
    featurizer per arm from the cached frame count, exactly as if it had been recomputed;
  * the value-of-computation target assembly, the online-state input assembly, the causal firing pass,
    the pooled referee scoring, the FLOP model, and the budget-point assembly (from artifact.py);
  * the three sealed controls (rate-matched-random, always-on, best-single) and the noisy-TV channel;
  * the matched-budget harness and the exact sign-flip statistics;
  * the committed SESOI (0.05) and the sign-flip plan, read from the sealed preregistrations. Nothing here
    changes the SESOI or any threshold.

What differs from the committed candidate: only the gate. Its architecture, initialization, online state,
inference path, and per-step training cost are byte-identical to the committed gate; its training loss adds
a within-clip determinantal spacing regularizer. Its strength is selected per seed on the val rooms over a
preregistered grid (val tuning, never a test-score read), the same way theta and the best-single threshold
are already tuned on val.

Honesty notes carried into the sealed artifact:
  * ``diversity_reg`` is an ADDITIONAL exploratory variant beyond the sealed four-variant E1 family
    (proof/STARSS23_ESCS_BED_VARIANTS.prereg.json). It reuses the identical sealed metric, direction, SESOI
    0.05, controls, referee, and sign-flip plan, and only inflates the variant family, so the preregistered
    multiplicity wall is if anything stronger. No single run can promote; the flags are hardcoded false.
  * activation_allowed=false, scientific_promotion=false, independent_scientific_confirmation=false.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from mop.beds.starss23 import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL  # noqa: E402
from mop.beds.starss23.artifact import (  # noqa: E402
    ARTIFACT_SCHEMA,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
    STAGE3_REQUIREMENT_ID,
    BedConfig,
    _assemble_inputs,
    _build_budget_points,
    _causal_fires,
    _pooled_score,
    _SeedRun,
    _voc_targets,
)
from mop.beds.starss23.controls import (  # noqa: E402
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from mop.beds.starss23.feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus  # noqa: E402
from mop.beds.starss23.featurizer import FLOPS_PER_FRAME, FrozenFeaturizer  # noqa: E402
from mop.beds.starss23.gate import FLOPS_PER_INFERENCE, OnlineState  # noqa: E402
from mop.beds.starss23.gate_diversity_reg import DEFAULT_SPACING_WINDOW, DiversityRegGate  # noqa: E402
from mop.beds.starss23.prereg import PREREGISTERED_SESOI_F1  # noqa: E402
from mop.beds.starss23.real_artifact import RealBedConfig, _real_noisy_tv_features  # noqa: E402
from mop.beds.starss23.schema import COLLAR_FRAMES, Clip, ClipSplit  # noqa: E402
from mop.ladder.ladder_contracts import (  # noqa: E402
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    mint_demonstration,
)
from mop.science.budget import (  # noqa: E402
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    run_matched_budget,
)
from mop.science.statistics import exact_sign_flip  # noqa: E402
from mop.substrate.events import canonical_bytes, canonical_sha256  # noqa: E402

VARIANT_ID = "diversity_reg"
VARIANT_PRODUCER_SCHEMA = "mop-starss23-escs-diversity-reg-producer/v1"
SPACING_WINDOW = DEFAULT_SPACING_WINDOW

# Preregistered before the run: the val-selection grid of spacing-regularizer strengths. All strictly
# positive, so the variant always applies the diagnosed-failure countermeasure; val onset-F1 at the
# operating budget selects the strength per seed (never a test-score read). Ties keep the smaller strength.
DIVERSITY_LAMBDA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

DEFAULT_OUT_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED_diversity_reg.json"
BASE_PREREG_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED.prereg.json"
VARIANTS_PREREG_PATH = REPO_ROOT / "proof" / "STARSS23_ESCS_BED_VARIANTS.prereg.json"

# Committed baseline reference at the operating budget (seed 0), from the sealed committed real null
# proof/STARSS23_ESCS_BED.json: the trained value-of-computation gate recovered 204 distinct-onset true
# positives against 237 for rate-matched-random. Carried for context only; the head-to-head test in this
# run is the variant against its own per-seed rate-matched-random control.
COMMITTED_BASELINE_TP_SEED0 = 204
COMMITTED_RANDOM_TP_SEED0 = 237


def _train_variant_gate(
    seed: int,
    split_train: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    config: BedConfig,
    diversity_lambda: float,
) -> tuple[DiversityRegGate, int]:
    """Train one diversity_reg gate on the train rooms. Returns the gate and the train-frame count.

    The training inputs and value-of-computation targets are assembled by the same reused helpers the
    committed producer uses; ``segment_lengths`` carries the per-clip frame counts so the spacing
    regularizer never couples two different train clips.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    segment_lengths: list[int] = []
    for clip in split_train:
        features = features_by_clip[clip.clip_id]
        inputs.append(_assemble_inputs(features))
        targets.append(_voc_targets(clip.onset_frames, clip.n_frames, window=config.voc_window))
        segment_lengths.append(int(clip.n_frames))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = DiversityRegGate(seed=seed, diversity_lambda=diversity_lambda, spacing_window=SPACING_WINDOW)
    gate.fit(
        x,
        y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
        segment_lengths=segment_lengths,
    )
    return gate, int(x.shape[0])


def _val_f1_at_operating_rate(
    gate: DiversityRegGate,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    operating_rate: float,
) -> float:
    """Score the gate's val onset-F1 at the operating firing budget. Reads only val labels, never test."""

    from mop.beds.starss23.referee import score_arm

    val_probs = np.concatenate(
        [_causal_fires(gate, features_by_clip[clip.clip_id], 0.5)[1] for clip in split.val]
    )
    theta = float(np.quantile(val_probs, 1.0 - operating_rate))
    scored = [
        (list(clip.onset_frames), _causal_fires(gate, features_by_clip[clip.clip_id], theta)[0])
        for clip in split.val
    ]
    return score_arm(scored, COLLAR_FRAMES).f1


def _select_diversity_lambda(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    config: BedConfig,
    operating_rate: float,
) -> tuple[float, DiversityRegGate, int, dict[str, float]]:
    """Select the spacing strength on val onset-F1 at the operating budget, over the preregistered grid.

    Val tuning only: no test-split score is read. Ties keep the smaller strength (the grid is ascending and
    a strength replaces the incumbent only on a strictly greater val F1), the conservative choice closest to
    the committed gate.
    """

    best_lambda = DIVERSITY_LAMBDA_GRID[0]
    best_gate: DiversityRegGate | None = None
    best_frames = 0
    best_f1 = -1.0
    val_f1_by_lambda: dict[str, float] = {}
    for diversity_lambda in DIVERSITY_LAMBDA_GRID:
        gate, train_frames = _train_variant_gate(
            seed, split.train, features_by_clip, config, diversity_lambda
        )
        val_f1 = _val_f1_at_operating_rate(gate, split, features_by_clip, operating_rate)
        val_f1_by_lambda[f"{diversity_lambda:g}"] = float(val_f1)
        if val_f1 > best_f1 + 1e-12:
            best_f1 = val_f1
            best_lambda = diversity_lambda
            best_gate = gate
            best_frames = train_frames
    assert best_gate is not None
    return best_lambda, best_gate, best_frames, val_f1_by_lambda


def _adjacency_fraction(fires_by_clip: list[list[int]]) -> tuple[float, int, int]:
    """Fraction of fires immediately adjacent (gap of one frame) to another fire in the same clip.

    Returns (adjacency_fraction, adjacent_fire_count, total_fire_count). This is the exact fire-spread
    diagnostic the committed null flagged: clustered fires waste budget because two fires inside one collar
    can score at most one distinct-onset true positive.
    """

    adjacent = 0
    total = 0
    for fires in fires_by_clip:
        fireset = set(fires)
        for frame in fires:
            total += 1
            if (frame - 1) in fireset or (frame + 1) in fireset:
                adjacent += 1
    fraction = adjacent / total if total > 0 else 0.0
    return fraction, adjacent, total


def _run_seed_variant(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: BedConfig,
    operating_density: float,
    operating_rate: float,
) -> tuple[_SeedRun, dict[str, Any]]:
    """Select the strength on val, then sweep the budget and score every arm on the fixed real test set."""

    from mop.beds.starss23.referee import score_arm

    selected_lambda, gate, train_frames, val_f1_by_lambda = _select_diversity_lambda(
        seed, split, features_by_clip, config, operating_rate
    )
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

    # Fire-spread diagnostic at the operating budget: adjacency fraction and distinct-onset true positives
    # for the variant and its rate-matched-random control, measured identically in the same run.
    candidate_fires_by_clip = [block["fires"][ARM_CANDIDATE] for block in operating["clips"]]
    rmr_fires_by_clip = [block["fires"][ARM_RATE_MATCHED_RANDOM] for block in operating["clips"]]
    cand_adj, cand_adj_n, cand_total = _adjacency_fraction(candidate_fires_by_clip)
    rmr_adj, rmr_adj_n, rmr_total = _adjacency_fraction(rmr_fires_by_clip)
    variant_detail = {
        "seed": seed,
        "operating_budget_id": operating_budget_id,
        "selected_diversity_lambda": float(selected_lambda),
        "val_f1_by_lambda": val_f1_by_lambda,
        "candidate_distinct_onset_tp": operating["arm_scores"][ARM_CANDIDATE]["tp"],
        "rate_matched_random_distinct_onset_tp": operating["arm_scores"][ARM_RATE_MATCHED_RANDOM]["tp"],
        "candidate_f1": operating["arm_scores"][ARM_CANDIDATE]["f1"],
        "rate_matched_random_f1": operating["arm_scores"][ARM_RATE_MATCHED_RANDOM]["f1"],
        "candidate_fire_count": cand_total,
        "rate_matched_random_fire_count": rmr_total,
        "candidate_adjacency_fraction": round(float(cand_adj), 12),
        "rate_matched_random_adjacency_fraction": round(float(rmr_adj), 12),
    }

    seed_run = _SeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
    )
    return seed_run, variant_detail


def _read_prereg_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_bytes().decode("utf-8"))
    except (ValueError, OSError):
        return None
    digest = body.get("canonical_sha256")
    return str(digest) if isinstance(digest, str) else None


def build_diversity_reg_artifact(
    *,
    timestamp: str,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    config: RealBedConfig | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the diversity_reg variant on the cached real corpus and assemble the sealed artifact body.

    Returns the byte-sealed body and the measured wall in nanoseconds (unsealed run provenance).
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = FrozenFeaturizer()

    corpus = load_cached_corpus(cache_root=cache_root)
    features_by_clip = corpus.features_by_clip
    split = corpus.split

    train_density = corpus.train_onset_density()
    n_test_clips = corpus.n_test_clips()
    n_test_onsets = corpus.n_test_onsets()
    n_test_frames = corpus.n_test_frames()
    if n_test_onsets == 0:
        raise SystemExit("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # The noisy-TV channel marginals are matched to the pooled real test content, as in the committed run.
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in split.test], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_SeedRun] = []
    variant_details: list[dict[str, Any]] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_run, variant_detail = _run_seed_variant(
            seed, split, features_by_clip, noise_features, bed_config, train_density, operating_rate
        )
        seed_runs.append(seed_run)
        variant_details.append(variant_detail)
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
    sesoi_f1 = float(PREREGISTERED_SESOI_F1)
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
        "prereg_canonical_sha256": _read_prereg_digest(BASE_PREREG_PATH),
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

    # Fire-spread diagnostic aggregated across seeds, versus the committed baseline reference.
    mean_candidate_tp = math.fsum(d["candidate_distinct_onset_tp"] for d in variant_details) / n_runs
    mean_rmr_tp = (
        math.fsum(d["rate_matched_random_distinct_onset_tp"] for d in variant_details) / n_runs
    )
    mean_candidate_adjacency = (
        math.fsum(d["candidate_adjacency_fraction"] for d in variant_details) / n_runs
    )
    mean_rmr_adjacency = (
        math.fsum(d["rate_matched_random_adjacency_fraction"] for d in variant_details) / n_runs
    )
    fire_spread_block = {
        "per_seed": variant_details,
        "mean_candidate_distinct_onset_tp": round(float(mean_candidate_tp), 6),
        "mean_rate_matched_random_distinct_onset_tp": round(float(mean_rmr_tp), 6),
        "mean_candidate_adjacency_fraction": round(float(mean_candidate_adjacency), 12),
        "mean_rate_matched_random_adjacency_fraction": round(float(mean_rmr_adjacency), 12),
        "committed_baseline_distinct_onset_tp_seed0": COMMITTED_BASELINE_TP_SEED0,
        "committed_rate_matched_random_distinct_onset_tp_seed0": COMMITTED_RANDOM_TP_SEED0,
        "note": (
            "distinct-onset true positives equal the one-to-one matched count at the operating budget; the "
            "committed null recovered 204 for the trained gate against 237 for rate-matched-random at seed "
            "0. Adjacency fraction is the share of fires immediately next to another fire in the same clip, "
            "the clustering the regularizer targets"
        ),
    }

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
            "variant_id": VARIANT_ID,
            "forcing_null": STAGE3_FORCING_NULL,
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one real run of an additional exploratory gate variant is a mechanics demonstration; "
                "scientific confirmation needs the independent verifier plus at least three bias-"
                "independent reproductions and cannot be self-certified"
            ),
        },
    )

    selected_lambdas = {str(d["seed"]): d["selected_diversity_lambda"] for d in variant_details}
    per_seed_train_frames = seed_runs[0].train_frames
    search_flops = int(
        len(DIVERSITY_LAMBDA_GRID)
        * n_runs
        * FLOPS_PER_INFERENCE
        * per_seed_train_frames
        * bed_config.epochs
        * 3
    )

    variant_block = {
        "variant_id": VARIANT_ID,
        "producer_schema": VARIANT_PRODUCER_SCHEMA,
        "hypothesis": (
            "a within-clip determinantal spacing regularizer added to the training loss pushes the gate to "
            "place the same firing budget on distinct onsets instead of clustering adjacently on high-"
            "energy regions, so it beats rate-matched-random at matched firing count"
        ),
        "mechanism": (
            "committed 264 -> 12 -> 1 architecture, identical initialization, online state, and inference "
            "path; the only change is a training loss term "
            "S = (lambda / n) * sum over within-clip pairs closer than the collar of kernel(distance) * "
            "p_i * p_j, a soft determinantal-point-process repulsion"
        ),
        "spacing_window_frames": SPACING_WINDOW,
        "spacing_window_ms": SPACING_WINDOW * 100,
        "spacing_window_rationale": (
            "anchored to the sealed referee collar; two fires closer than the collar can match at most one "
            "distinct onset under one-to-one matching, so co-firing inside the collar is the wasted budget "
            "the regularizer prices"
        ),
        "diversity_lambda_grid": list(DIVERSITY_LAMBDA_GRID),
        "strength_selection_rule": (
            "per seed, the grid strength with the highest val onset-F1 at the operating budget; val tuning "
            "only, never a test-score read; ties keep the smaller strength"
        ),
        "selected_diversity_lambda_by_seed": selected_lambdas,
        "hyperparameter_search_flops": search_flops,
        "hyperparameter_search_note": (
            "the per-arm sealed FLOP ledger charges one C_train (the selected strength's training), exactly "
            "as the committed candidate is charged and as val theta-tuning and best-single tuning are "
            "already uncharged in this bed. This field records the honest full development cost of the "
            "val-selection grid ("
            f"{len(DIVERSITY_LAMBDA_GRID)} strengths times {n_runs} seeds) as unsealed provenance"
        ),
        "preregistration_status": (
            "diversity_reg is an ADDITIONAL exploratory variant beyond the sealed four-variant E1 family "
            "(proof/STARSS23_ESCS_BED_VARIANTS.prereg.json: refractory_nms, flux_novelty_target, "
            "energy_whitened_features, recency_gap_penalty). It reuses the identical sealed metric, "
            "direction, SESOI 0.05, controls, referee, and sign-flip plan and changes no threshold. Adding "
            "a fifth variant only inflates the variant family, so the preregistered Bonferroni multiplicity "
            "wall is if anything stronger; no single run, and no run at n equals 5 across this family, can "
            "promote"
        ),
        "variants_prereg_canonical_sha256": _read_prereg_digest(VARIANTS_PREREG_PATH),
        "fire_spread_diagnostic": fire_spread_block,
    }

    truncations = list(corpus.truncations)
    dropped_onsets = sum(int(t.get("dropped_onsets_past_end", 0)) for t in truncations)
    capped_clips = sum(1 for t in truncations if t.get("capped_by_max_frames"))

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
        "variant_id": VARIANT_ID,
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
        "variant": variant_block,
        "real_corpus": {
            "producer_schema": VARIANT_PRODUCER_SCHEMA,
            "feature_cache_key": corpus.cache_key,
            "feature_cache_dir": str(corpus.cache_dir),
            "featurizer_digest": corpus.featurizer_digest,
            "foa_root": corpus.foa_root,
            "metadata_root": corpus.metadata_root,
            "n_clips": len(corpus.clips),
            "split_rooms": dict(split.detail),
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": n_test_frames,
            "train_onset_density": round(float(train_density), 12),
            "operating_firing_fraction": round(float(operating_rate), 12),
            "truncation": {
                "clips_capped_by_max_frames": capped_clips,
                "onsets_dropped_past_audio_end": dropped_onsets,
                "max_frames": corpus.max_frames,
                "per_clip": truncations,
            },
        },
        "prereg": {
            "base_prereg_path": str(BASE_PREREG_PATH.relative_to(REPO_ROOT)),
            "base_prereg_canonical_sha256": _read_prereg_digest(BASE_PREREG_PATH),
            "variants_prereg_path": str(VARIANTS_PREREG_PATH.relative_to(REPO_ROOT)),
            "variants_prereg_canonical_sha256": _read_prereg_digest(VARIANTS_PREREG_PATH),
            "sesoi_f1": sesoi_f1,
            "provisional": False,
            "sesoi_unchanged": True,
            "written_before_test_scores": True,
            "note": (
                "the SESOI 0.05 and the sign-flip plan are the committed preregistered values, reused "
                "unchanged; this variant run reads them and changes no threshold"
            ),
        },
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)
    return body, int(measured_wall_ns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the STARSS23 ESCS diversity_reg gate variant.")
    parser.add_argument("out_path", nargs="?", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--timestamp", default="2026-07-17T00:00:00Z")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    body, measured_wall_ns = build_diversity_reg_artifact(
        timestamp=args.timestamp, cache_root=args.cache_root
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(canonical_bytes(body))

    per_budget_rows = body["harness"]["per_budget_candidate_vs_rate_matched_random"]
    fsd = body["variant"]["fire_spread_diagnostic"]
    print(f"wrote {out_path}")
    print(
        f"variant={VARIANT_ID} source_kind=real rights_clean=true verdict={body['verdict']} "
        f"seal={body['seal']}"
    )
    print(
        f"dominates={body['harness']['candidate_strictly_dominates_rate_matched_random']} "
        f"mean_delta={body['stats']['t_obs']:.6f} one_sided_p={body['stats']['one_sided_p']} "
        f"sesoi_f1={body['stats']['sesoi_f1']} "
        f"exceeds_sesoi={body['stats']['mean_delta_exceeds_sesoi']} "
        f"noisy_tv_at_chance={body['controls']['noisy_tv_at_chance']}"
    )
    print(f"per_seed_deltas={body['stats']['deltas']}")
    print(f"selected_diversity_lambda_by_seed={body['variant']['selected_diversity_lambda_by_seed']}")
    for row in per_budget_rows:
        print(
            f"  {row['budget_id']}: candidate {row['candidate_mean_f1']:.6f} vs "
            f"rate_matched_random {row['rate_matched_random_mean_f1']:.6f} "
            f"delta {row['delta_mean_f1']:.6f} "
            f"beats={row['candidate_strictly_beats_rate_matched_random']}"
        )
    print(
        f"fire_spread: mean_candidate_tp={fsd['mean_candidate_distinct_onset_tp']} "
        f"mean_rmr_tp={fsd['mean_rate_matched_random_distinct_onset_tp']} "
        f"(committed baseline seed0 tp=204 vs random 237); "
        f"mean_candidate_adjacency={fsd['mean_candidate_adjacency_fraction']:.4f} "
        f"mean_rmr_adjacency={fsd['mean_rate_matched_random_adjacency_fraction']:.4f}"
    )
    print(f"measured_wall_ns={measured_wall_ns} (unsealed run provenance)")
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real run of an exploratory variant; mechanics only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
