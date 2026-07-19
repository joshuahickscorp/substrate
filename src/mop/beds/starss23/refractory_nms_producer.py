"""Real-data producer for the E1 gate variant "refractory_nms" on the cached STARSS23 corpus.

This is a net-new, additive component. It runs the refractory_nms gate variant end to end on the REAL,
MIT-licensed STARSS23 FOA subset and assembles a sealed ``proof/STARSS23_ESCS_BED_refractory_nms.json``
with the same shape as the committed ``proof/STARSS23_ESCS_BED.json``. It changes NONE of the sealed
scoring logic: the referee, the matched-budget harness and its FLOP accounting, the exact sign-flip
statistics, and every control are imported unchanged, and the value-of-computation training-target
assembly, the causal p_fire pass, the FLOP model, and the budget-point assembly are imported unchanged
from the committed producer so the variant lane scores through the identical machinery. The ONLY thing
that differs from the committed real run is the firing-selection policy: the candidate commits fires by
collar-width refractory NMS on the same p_fire trace (see ``gate_refractory_nms``).

Performance: the deterministic featurizer is the dominant, gate-independent cost, so it is never
recomputed here. The producer consumes the shared feature cache (``feature_cache.load_cached_corpus``),
which featurized the room-disjoint subset exactly once. The FLOP ledger still charges the featurizer per
arm from the cache's honest per-frame count, so caching is a wall-clock optimization, never a budget cut.

The SESOI and the whole analysis plan are read from the already-sealed variant preregistration
``proof/STARSS23_ESCS_BED_VARIANTS.prereg.json``; this producer NEVER rebuilds or reweakens it. The
verdict is a mechanics outcome only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single run at n equals 5 across the
four-variant family can never promote.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import math
import time
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
    build_budget_points,
    noise_control_summary,
    run_matched_budget,
)
from mop.science.statistics import exact_sign_flip, sign_flip_payload

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
    _assemble_inputs,
    _causal_fires,
    _flop_model,
    _pooled_score,
    _SeedRun,
    _voc_targets,
)
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .experiments import ONSET_BUDGET_POLICY
from .feature_cache import DEFAULT_CACHE_ROOT, CachedCorpus, load_cached_corpus
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate import FLOPS_PER_INFERENCE, CandidateGate, OnlineState
from .gate_refractory_nms import (
    DEFAULT_WINDOW_FRAMES,
    RefractoryNmsGate,
    tune_theta_for_rate,
)
from .gate_variants_prereg import DEFAULT_VARIANTS_PREREG_PATH, GATE_VARIANTS, VARIANTS_PREREG_SCHEMA
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealBedConfig,
    _real_noisy_tv_features,
)
from .referee import score_arm
from .schema import COLLAR_FRAMES, ClipSplit

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-refractory-nms/v1"
VARIANT_ID = "refractory_nms"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_refractory_nms.json")


class RefractoryNmsRefusal(ValueError):
    """Raised when the refractory_nms producer cannot assemble a well-formed sealed artifact."""


def _variant_hypothesis() -> str:
    for entry in GATE_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise RefractoryNmsRefusal(f"variant {VARIANT_ID!r} is not in the sealed variant family")


def _read_sealed_variants_prereg(path: str | Path = DEFAULT_VARIANTS_PREREG_PATH) -> dict[str, Any]:
    """Read the already-sealed variant preregistration. Never rebuilds or reweakens it."""

    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise RefractoryNmsRefusal(
            f"the sealed variant preregistration {prereg_path} is missing; seal it before the run"
        )
    body = json.loads(prereg_path.read_bytes().decode("utf-8"))
    if body.get("schema") != VARIANTS_PREREG_SCHEMA:
        raise RefractoryNmsRefusal(f"unexpected variant prereg schema {body.get('schema')!r}")
    ids = [entry["variant_id"] for entry in body.get("variants", [])]
    if VARIANT_ID not in ids:
        raise RefractoryNmsRefusal(f"{VARIANT_ID!r} is not preregistered in {prereg_path}")
    return body


# ---------------------------------------------------------------------------
# Training: identical to the committed producer, only the gate class differs.
# ---------------------------------------------------------------------------


def _train_refractory_gate(
    seed: int,
    split_train: Any,
    features_by_clip: dict[str, np.ndarray],
    epochs: int,
    learning_rate: float,
    ponder_lambda: float,
    voc_window: int,
    window: int,
) -> tuple[RefractoryNmsGate, int]:
    """Train the variant gate. Byte-identical training to the committed gate; only the class differs.

    Uses the committed producer's value-of-computation target assembly (``_assemble_inputs`` with the
    label-free online-state pass and ``_voc_targets``) unchanged. ``RefractoryNmsGate`` subclasses
    ``CandidateGate`` with an identical constructor and ``fit``, so the trained weights are byte-identical
    to the committed gate at the same seed. The refractory NMS only changes the firing selection later.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in split_train:
        features = features_by_clip[clip.clip_id]
        inputs.append(_assemble_inputs(features))
        targets.append(_voc_targets(clip.onset_frames, clip.n_frames, window=voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = RefractoryNmsGate(seed=seed, window=window)
    gate.fit(x, y, epochs=epochs, learning_rate=learning_rate, ponder_lambda=ponder_lambda)
    return gate, int(x.shape[0])


# ---------------------------------------------------------------------------
# Fire-spread diagnostics: adjacency and distinct-onset true positives.
# ---------------------------------------------------------------------------


def _adjacency_fraction(clip_fire_lists: list[list[int]], collar: int = COLLAR_FRAMES) -> float:
    """Pooled fraction of fires that have another fire within ``collar`` frames in the same clip.

    This is the clustering diagnostic: the committed null clustered roughly 42 percent of its fires
    adjacently on high-energy regions. A fire is "adjacent" if a distinct fire on the same clip lands
    within the DCASE collar of it. Refractory NMS separates committed fires by more than the collar, so
    its adjacency fraction is zero by construction.
    """

    total = 0
    adjacent = 0
    for fires in clip_fire_lists:
        ordered = sorted(fires)
        total += len(ordered)
        for index, frame in enumerate(ordered):
            near_prev = index > 0 and frame - ordered[index - 1] <= collar
            near_next = index < len(ordered) - 1 and ordered[index + 1] - frame <= collar
            if near_prev or near_next:
                adjacent += 1
    return adjacent / total if total > 0 else 0.0


def _arm_spread(
    clip_gt_and_fires: list[tuple[list[int], list[int]]], collar: int = COLLAR_FRAMES
) -> dict[str, Any]:
    """Pooled fire count, adjacency fraction, and distinct-onset true positives for one arm and seed."""

    fire_lists = [fires for _gt, fires in clip_gt_and_fires]
    score = score_arm(clip_gt_and_fires, collar)
    total_fires = sum(len(fires) for fires in fire_lists)
    return {
        "fires": total_fires,
        "adjacency_fraction": round(_adjacency_fraction(fire_lists, collar), 12),
        "distinct_onset_tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
    }
# ---------------------------------------------------------------------------
# Per-seed run: variant candidate plus the three controls, and a committed-gate reference.
# ---------------------------------------------------------------------------


def _run_seed_variant(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: Any,
    operating_density: float,
    window: int,
) -> tuple[_SeedRun, dict[str, Any]]:
    """Run one paired seed of the variant, and reproduce the committed gate at its operating point.

    The variant candidate commits fires by refractory NMS on its p_fire trace; the rate-matched-random
    control matches the committed count per clip and seed; always-on and best-single are the committed
    controls unchanged. The threshold per swept budget is tuned so the POST-NMS val firing fraction hits
    the swept target rate, so the variant spends the same firing budget spread rather than clustered. The
    committed gate (raw threshold firing) is reproduced at the same operating point purely as a
    fire-spread reference, so this artifact carries the 204-true-positive / 42-percent-adjacency anchor of
    the committed null internally.
    """

    gate, train_frames = _train_refractory_gate(
        seed,
        split.train,
        features_by_clip,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ponder_lambda=config.ponder_lambda,
        voc_window=config.voc_window,
        window=window,
    )
    total_frames = int(sum(clip.n_frames for clip in split.test))

    # Reference val p_fire traces at a neutral threshold, the distribution the budget grid is cut from.
    val_prob_traces = [gate.causal_probs(features_by_clip[clip.clip_id], 0.5) for clip in split.val]
    best_single = BestSingleControl.tuned(
        [(features_by_clip[clip.clip_id], list(clip.onset_frames)) for clip in split.val]
    )

    per_budget: dict[str, dict[str, Any]] = {}
    variant_test_fires: dict[str, dict[str, list[list[int]]]] = {}
    for rate in config.target_rates:
        theta = tune_theta_for_rate(val_prob_traces, rate, window)
        budget_id = f"rate_{rate:.2f}"

        val_scored = []
        for clip in split.val:
            fires = gate.refractory_fires(features_by_clip[clip.clip_id], theta, window)
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
            candidate_fires = gate.refractory_fires(features, theta, window)
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
        variant_test_fires[budget_id] = {
            kind: [fires for _gt, fires in pairs] for kind, pairs in arm_clip_scores.items()
        }

    # Preregistered operating point: the swept budget whose rate is closest to the train onset density.
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

    # noisy-TV at the operating threshold, under the identical refractory NMS firing policy.
    operating_theta = operating["theta"]
    base_rate = operating["firings"][ARM_CANDIDATE] / max(1, total_frames)
    noise_fires = gate.refractory_fires(noise_features, operating_theta, window)
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
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

    # Fire-spread diagnostic at the operating point: the variant, and the committed gate reproduced.
    op_gt = [list(clip.onset_frames) for clip in split.test]
    variant_candidate = list(
        zip(op_gt, variant_test_fires[operating_budget_id][ARM_CANDIDATE], strict=True)
    )
    variant_rmr = list(
        zip(op_gt, variant_test_fires[operating_budget_id][ARM_RATE_MATCHED_RANDOM], strict=True)
    )
    base_ref = _committed_gate_reference(seed, split, features_by_clip, config, operating_density)
    diagnostic = {
        "seed": seed,
        "operating_budget_id": operating_budget_id,
        "variant": {
            "candidate": _arm_spread(variant_candidate),
            "rate_matched_random": _arm_spread(variant_rmr),
        },
        "committed_gate_reference": base_ref,
    }
    return seed_run, diagnostic


def _committed_gate_reference(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    config: Any,
    operating_density: float,
) -> dict[str, Any]:
    """Reproduce the committed gate's operating-point firing (raw threshold) as a fire-spread anchor.

    This trains the committed ``CandidateGate`` (byte-identical weights to the variant) and fires it with
    the committed raw-threshold rule at the committed operating point, so the artifact carries the
    committed null's clustered-firing fingerprint (its distinct-onset true positives and its adjacency
    fraction) reproduced from the cached corpus, not merely quoted.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in split.train:
        features = features_by_clip[clip.clip_id]
        inputs.append(_assemble_inputs(features))
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

    val_probs = np.concatenate(
        [_causal_fires(gate, features_by_clip[clip.clip_id], 0.5)[1] for clip in split.val]
    )
    rates = {rate: float(np.quantile(val_probs, 1.0 - rate)) for rate in config.target_rates}
    operating_rate = min(rates, key=lambda r: abs(r - operating_density))
    theta = rates[operating_rate]

    candidate_pairs: list[tuple[list[int], list[int]]] = []
    rmr_pairs: list[tuple[list[int], list[int]]] = []
    for clip in split.test:
        gt = list(clip.onset_frames)
        candidate_fires, _ = _causal_fires(gate, features_by_clip[clip.clip_id], theta)
        rmr_fires = rate_matched_random_fires(
            candidate_fires, clip.n_frames, seed=seed, clip_id=clip.clip_id
        )
        candidate_pairs.append((gt, candidate_fires))
        rmr_pairs.append((gt, rmr_fires))
    return {
        "operating_rate": operating_rate,
        "candidate": _arm_spread(candidate_pairs),
        "rate_matched_random": _arm_spread(rmr_pairs),
    }


# ---------------------------------------------------------------------------
# Assemble and seal the variant artifact.
# ---------------------------------------------------------------------------


def build_refractory_nms_artifact(
    *,
    timestamp: str,
    corpus: CachedCorpus | None = None,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    window: int = DEFAULT_WINDOW_FRAMES,
    config: RealBedConfig | None = None,
    variants_prereg_path: str | Path = DEFAULT_VARIANTS_PREREG_PATH,
) -> ArtifactResult:
    """Run the refractory_nms variant on the cached real corpus and assemble the sealed artifact.

    The SESOI and the analysis plan come from the already-sealed variant preregistration; this producer
    never rebuilds or reweakens it. ``timestamp`` is passed by the caller and never read from the wall
    clock inside a sealed body. The frozen featurizer is never recomputed: the corpus is loaded from the
    shared feature cache, and the FLOP ledger still charges the featurizer per arm from the cache count.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    corpus = corpus or load_cached_corpus(cache_root=cache_root)
    split = corpus.split
    features_by_clip = corpus.features_by_clip

    prereg = _read_sealed_variants_prereg(variants_prereg_path)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])
    prereg_digest = str(prereg["canonical_sha256"])

    # Structural facts (label-only), used for provenance and the operating-point rule.
    train_density = corpus.train_onset_density()
    n_test_clips = corpus.n_test_clips()
    n_test_onsets = corpus.n_test_onsets()
    n_test_frames = corpus.n_test_frames()
    if n_test_onsets == 0:
        raise RefractoryNmsRefusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # noisy-TV marginals: match the injected white-noise channel to the real test feature marginals.
    featurizer = FrozenFeaturizer()
    pooled_test_features = np.concatenate(
        [features_by_clip[clip.clip_id] for clip in split.test], axis=0
    )
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_SeedRun] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_run, diagnostic = _run_seed_variant(
            seed, split, features_by_clip, noise_features, bed_config, train_density, window
        )
        seed_runs.append(seed_run)
        diagnostics.append(diagnostic)
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
    beats_random = bool(sign_flip.one_sided_significant and mean_delta_exceeds_sesoi)
    stats_block = sign_flip_payload(
        sign_flip, deltas, sesoi_key="sesoi_f1", sesoi=sesoi_f1,
        exceeds_sesoi=mean_delta_exceeds_sesoi, provisional=False,
        prereg_digest=prereg_digest, extra={"beats_rate_matched_random": beats_random},
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

    spread_block = _assemble_spread_diagnostic(diagnostics)

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
            "variant_id": VARIANT_ID,
            "forcing_null": STAGE3_FORCING_NULL,
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one real run of one gate variant is a mechanics outcome; scientific confirmation needs "
                "the independent verifier plus at least three bias-independent reproductions and, for this "
                "four-variant family at n equals 5, cannot clear family-wise significance at all"
            ),
        },
    )

    body: dict[str, Any] = {
        "schema": VARIANT_ARTIFACT_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "variant_id": VARIANT_ID,
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
        "beats_rate_matched_random": beats_random,
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
            "feature_cache_key": corpus.cache_key,
            "note": "featurized once and cached; the FLOP ledger charges it per arm from the cache count",
        },
        "gate": {
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
        "variant": {
            "variant_id": VARIANT_ID,
            "hypothesis": _variant_hypothesis(),
            "firing_policy": "collar-width refractory non-maximum suppression on the p_fire trace",
            "window_frames": int(window),
            "window_ms": int(window) * 100,
            "only_firing_policy_differs": True,
            "trained_weights_identical_to_committed_gate": True,
            "variants_prereg_path": str(Path(variants_prereg_path)),
            "variants_prereg_canonical_sha256": prereg_digest,
            "fire_spread_diagnostic": spread_block,
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing,
        },
        "real_corpus": {
            "producer_schema": REAL_PRODUCER_SCHEMA,
            "variant_producer_schema": VARIANT_ARTIFACT_SCHEMA,
            "foa_root": str(Path(DEFAULT_FOA_ROOT)),
            "metadata_root": str(Path(DEFAULT_METADATA_ROOT)),
            "feature_cache_key": corpus.cache_key,
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
            "path": str(Path(variants_prereg_path)),
            "canonical_sha256": prereg_digest,
            "sesoi_f1": sesoi_f1,
            "provisional": False,
            "written_before_test_scores": True,
            "rebuilt_by_this_producer": False,
        },
        "demonstration_receipt": receipt,
    }
    return finalize_artifact(
        body,
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
            "spread": spread_block,
        },
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _assemble_spread_diagnostic(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the per-seed fire-spread diagnostics: variant vs committed-gate reference."""

    def _summary(path: tuple[str, ...]) -> dict[str, Any]:
        per_seed = []
        for diag in diagnostics:
            node = diag
            for key in path:
                node = node[key]
            per_seed.append(node)
        return {
            "per_seed_fires": [row["fires"] for row in per_seed],
            "per_seed_adjacency_fraction": [row["adjacency_fraction"] for row in per_seed],
            "per_seed_distinct_onset_tp": [row["distinct_onset_tp"] for row in per_seed],
            "mean_fires": round(_mean([row["fires"] for row in per_seed]), 6),
            "mean_adjacency_fraction": round(_mean([row["adjacency_fraction"] for row in per_seed]), 12),
            "mean_distinct_onset_tp": round(_mean([row["distinct_onset_tp"] for row in per_seed]), 6),
        }

    return {
        "definition": (
            "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of another "
            "fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one referee true "
            "positives at the operating budget. The committed-gate reference reproduces the committed "
            "null's clustered raw-threshold firing at its operating point from the same cached corpus."
        ),
        "collar_frames": COLLAR_FRAMES,
        "variant_candidate": _summary(("variant", "candidate")),
        "variant_rate_matched_random": _summary(("variant", "rate_matched_random")),
        "committed_gate_candidate": _summary(("committed_gate_reference", "candidate")),
        "committed_gate_rate_matched_random": _summary(
            ("committed_gate_reference", "rate_matched_random")
        ),
        "committed_null_seed0_anchor": {
            "candidate_distinct_onset_tp": 204,
            "rate_matched_random_distinct_onset_tp": 237,
            "candidate_adjacency_fraction_approx": 0.42,
            "source": "docs/mixture_of_perspectives/26_escs_starss23_bed.md",
        },
        "per_seed": diagnostics,
    }
