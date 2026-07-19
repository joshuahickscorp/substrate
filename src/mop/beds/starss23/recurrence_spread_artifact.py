"""Real-data producer for the ``recurrence_spread`` E1 gate variant.

This is a net-new, additive component. It runs the recurrence_spread variant end to end on the REAL,
MIT-licensed STARSS23 FOA corpus by reading the SHARED feature cache (never re-featurizing), and assembles
a sealed ``proof/STARSS23_ESCS_BED_recurrence_spread.json`` with the same shape as the committed artifact.
It changes none of the sealed scoring logic: the referee, the matched-budget harness and its FLOP
accounting, the exact sign-flip statistics, and every control (rate-matched-random, always-on,
best-single, noisy-TV) are imported UNCHANGED, and the committed value-of-computation signal head is
trained UNCHANGED through ``artifact._train_gate``. The only thing that differs from the committed null is
the firing policy of ``RecurrenceSpreadGate``.

The dominant cost, the deterministic frozen featurizer, is computed once and cached
(``feature_cache.load_cached_corpus``); this variant reuses that cache, so a run is featurize-free. The
FLOP ledger still charges the featurizer per arm honestly from the cache's frame count, so caching is a
wall-clock optimization only, not a budget reduction.

Preregistration: the SESOI (0.05), the metric, the direction, the sign-flip plan, and the multiplicity
wall are the sealed ``proof/STARSS23_ESCS_BED_VARIANTS.prereg.json`` (built by ``gate_variants_prereg``),
read here and never re-sealed or weakened. recurrence_spread operationalizes that prereg's recurrence-aware
family. The verdict is a mechanics demonstration only: ``activation_allowed``, ``scientific_promotion``,
and ``independent_scientific_confirmation`` are hardcoded false, and a single run can never be
scientifically confirmed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

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
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .artifact import (
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
    STAGE3_REQUIREMENT_ID,
    BedConfig,
    _pooled_score,
    _train_gate,
)
from .controls import (
    BestSingleControl,
    always_on_fires,
    at_chance,
    rate_matched_random_fires,
)
from .feature_cache import DEFAULT_CACHE_ROOT, CachedCorpus, load_cached_corpus
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .gate import OnlineState
from .gate_recurrence_spread import (
    FLOPS_PER_INFERENCE_SPREAD,
    N_SPREAD_PARAMS,
    REFRACTORY_FRAMES,
    VARIANT_ID,
    RecurrenceSpreadGate,
)
from .harness import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    Arm,
    ArmSeedResult,
    BudgetPoint,
    FlopModel,
    run_matched_budget,
)
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealBedConfig,
    _real_noisy_tv_features,
)
from .referee import score_arm
from .schema import COLLAR_FRAMES, ClipSplit
from .stats import exact_sign_flip

RECURRENCE_SPREAD_SCHEMA = "mop-starss23-escs-bed-recurrence-spread/v1"
DEFAULT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_recurrence_spread.json")
DEFAULT_VARIANTS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_VARIANTS.prereg.json")

# The base committed null's seed-0 distinct-onset true positives, quoted for the fire-spread diagnostic.
BASE_NULL_DISTINCT_TP = 204
RANDOM_DISTINCT_TP = 237


class RecurrenceSpreadRefusal(ValueError):
    """Raised when the recurrence_spread producer cannot assemble a well-formed sealed artifact."""


# ---------------------------------------------------------------------------
# Sealed variant preregistration lookup (SESOI and digest). Never re-sealed here.
# ---------------------------------------------------------------------------


def _read_variants_prereg(path: str | Path = DEFAULT_VARIANTS_PREREG_PATH) -> dict[str, Any]:
    """Load the sealed E1 variants preregistration and return the fields this run is bound to."""

    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise RecurrenceSpreadRefusal(
            f"the sealed variants prereg {prereg_path} is missing; seal it before running the variant"
        )
    body = json.loads(prereg_path.read_bytes().decode("utf-8"))
    sesoi = body.get("sesoi", {}).get("sesoi_f1")
    if not isinstance(sesoi, (int, float)):
        raise RecurrenceSpreadRefusal("variants prereg is missing a numeric sesoi_f1")
    return {
        "path": str(prereg_path),
        "canonical_sha256": body.get("canonical_sha256"),
        "sesoi_f1": float(sesoi),
        "min_one_sided_p": float(body["sign_flip_test_plan"]["min_one_sided_p"]),
        "operating_firing_fraction": float(
            body["sesoi"]["cost_benefit"]["operating_firing_fraction"]
        ),
    }


# ---------------------------------------------------------------------------
# Per-seed run on the fixed real cached split.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SpreadSeedRun:
    seed: int
    total_frames: int
    train_frames: int
    gate_params: int
    signal_train_flops: int
    search_flops: int
    total_train_flops: int
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    per_seed_block: dict[str, Any]
    noisy_tv: dict[str, Any]
    spread_report: dict[str, Any]


def _run_seed_spread(
    seed: int,
    split: ClipSplit,
    features_by_clip: dict[str, np.ndarray],
    noise_features: np.ndarray,
    config: BedConfig,
    operating_density: float,
    operating_rate: float,
) -> _SpreadSeedRun:
    """Train the signal head and the spreading weights for one seed, sweep the budget, score every arm."""

    signal_gate, train_frames = _train_gate(seed, split.train, features_by_clip, config)
    gate = RecurrenceSpreadGate(signal_gate=signal_gate, rho=operating_rate)
    train_clips = [(features_by_clip[c.clip_id], list(c.onset_frames)) for c in split.train]
    spread_report = gate.fit_spread(train_clips, operating_rate)

    total_frames = int(sum(clip.n_frames for clip in split.test))
    signal_train_flops = gate.signal_training_flops(train_frames, config.epochs)
    total_train_flops = signal_train_flops + int(gate.search_flops)

    val_probs = np.concatenate(
        [gate.causal_pass(features_by_clip[clip.clip_id], 0.5)[1] for clip in split.val]
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
            fires, _ = gate.causal_pass(features_by_clip[clip.clip_id], theta)
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
            candidate_fires, _ = gate.causal_pass(features, theta)
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
    noise_fires, _ = gate.causal_pass(noise_features, operating_theta)
    noise_rate = len(noise_fires) / noise_features.shape[0]
    noisy_tv = {
        "firing_rate_on_noise": round(float(noise_rate), 12),
        "base_rate": round(float(base_rate), 12),
        "at_chance": at_chance(min(1.0, noise_rate), min(1.0, base_rate)),
        "n_noise_frames": int(noise_features.shape[0]),
    }

    return _SpreadSeedRun(
        seed=seed,
        total_frames=total_frames,
        train_frames=train_frames,
        gate_params=gate.n_params(),
        signal_train_flops=signal_train_flops,
        search_flops=int(gate.search_flops),
        total_train_flops=total_train_flops,
        per_budget=per_budget,
        operating_budget_id=operating_budget_id,
        per_seed_block=per_seed_block,
        noisy_tv=noisy_tv,
        spread_report=spread_report.payload(),
    )


# ---------------------------------------------------------------------------
# FLOP model and budget points (spread-inclusive per-frame inference, honest C_train).
# ---------------------------------------------------------------------------


def _flop_model_spread(
    kind: str, total_frames: int, total_train_flops: int
) -> FlopModel:
    """Full-lifecycle FLOP model. Candidate and rate-matched-random share the spread-inclusive infer."""

    featurize = FLOPS_PER_FRAME * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = FLOPS_PER_INFERENCE_SPREAD * total_frames if runs_gate else 0
    train = total_train_flops if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        train_flops=train,
    )


def _build_budget_points(seed_runs: list[_SpreadSeedRun]) -> list[BudgetPoint]:
    total_frames = seed_runs[0].total_frames
    total_train_flops = seed_runs[0].total_train_flops
    params = {ARM_CANDIDATE: seed_runs[0].gate_params}
    budget_points: list[BudgetPoint] = []
    for budget_id in seed_runs[0].per_budget:
        arms: dict[str, Arm] = {}
        for kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE):
            seed_results = tuple(
                ArmSeedResult(
                    seed=run.seed,
                    f1=run.per_budget[budget_id]["arm_scores"][kind]["f1"],
                    firings=run.per_budget[budget_id]["firings"][kind],
                )
                for run in seed_runs
            )
            arms[kind] = Arm(
                name=f"{kind}@{budget_id}",
                kind=kind,
                total_frames=total_frames,
                params=params.get(kind, 0),
                flop_model=_flop_model_spread(kind, total_frames, total_train_flops),
                seed_results=seed_results,
            )
        budget_points.append(
            BudgetPoint(
                budget_id=budget_id,
                candidate=arms[ARM_CANDIDATE],
                rate_matched_random=arms[ARM_RATE_MATCHED_RANDOM],
                always_on=arms[ARM_ALWAYS_ON],
                best_single=arms[ARM_BEST_SINGLE],
            )
        )
    return budget_points


# ---------------------------------------------------------------------------
# Fire-spread diagnostic: adjacency fraction and distinct-onset true positives.
# ---------------------------------------------------------------------------


def _adjacency_fraction(fires_per_clip: list[list[int]], window: int = COLLAR_FRAMES) -> dict[str, Any]:
    """Fraction of fires whose nearest other fire in the same clip is within ``window`` frames."""

    total = 0
    adjacent = 0
    for fires in fires_per_clip:
        ordered = sorted(fires)
        for i, frame in enumerate(ordered):
            total += 1
            near_prev = i > 0 and (frame - ordered[i - 1]) <= window
            near_next = i < len(ordered) - 1 and (ordered[i + 1] - frame) <= window
            if near_prev or near_next:
                adjacent += 1
    return {
        "n_fires": total,
        "n_adjacent": adjacent,
        "adjacency_fraction": round(adjacent / total, 12) if total else 0.0,
        "window_frames": window,
    }


def _fire_spread_diagnostic(seed_runs: list[_SpreadSeedRun]) -> dict[str, Any]:
    """Adjacency fraction and distinct-onset TP for the candidate and the control at the operating budget."""

    per_seed: list[dict[str, Any]] = []
    cand_adj: list[float] = []
    rmr_adj: list[float] = []
    cand_tp: list[int] = []
    rmr_tp: list[int] = []
    for run in seed_runs:
        block = run.per_seed_block
        cand_fires = [clip["fires"][ARM_CANDIDATE] for clip in block["clips"]]
        rmr_fires = [clip["fires"][ARM_RATE_MATCHED_RANDOM] for clip in block["clips"]]
        cand_adj_block = _adjacency_fraction(cand_fires)
        rmr_adj_block = _adjacency_fraction(rmr_fires)
        cand_distinct_tp = int(block["arm_scores"][ARM_CANDIDATE]["tp"])
        rmr_distinct_tp = int(block["arm_scores"][ARM_RATE_MATCHED_RANDOM]["tp"])
        cand_adj.append(cand_adj_block["adjacency_fraction"])
        rmr_adj.append(rmr_adj_block["adjacency_fraction"])
        cand_tp.append(cand_distinct_tp)
        rmr_tp.append(rmr_distinct_tp)
        per_seed.append(
            {
                "seed": run.seed,
                "candidate_adjacency": cand_adj_block,
                "rate_matched_random_adjacency": rmr_adj_block,
                "candidate_distinct_onset_tp": cand_distinct_tp,
                "rate_matched_random_distinct_onset_tp": rmr_distinct_tp,
            }
        )
    n = len(seed_runs)
    return {
        "window_frames": COLLAR_FRAMES,
        "base_null_distinct_onset_tp_seed0": BASE_NULL_DISTINCT_TP,
        "rate_matched_random_distinct_onset_tp_seed0": RANDOM_DISTINCT_TP,
        "mean_candidate_adjacency_fraction": round(math.fsum(cand_adj) / n, 12),
        "mean_rate_matched_random_adjacency_fraction": round(math.fsum(rmr_adj) / n, 12),
        "mean_candidate_distinct_onset_tp": round(math.fsum(cand_tp) / n, 6),
        "mean_rate_matched_random_distinct_onset_tp": round(math.fsum(rmr_tp) / n, 6),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Assemble and seal the recurrence_spread artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecurrenceSpreadArtifact:
    """The assembled sealed variant artifact plus the mechanics-only demonstration receipt."""

    artifact: dict[str, Any]
    verdict: str
    detail: dict[str, Any]

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def build_recurrence_spread_artifact(
    *,
    timestamp: str,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    variants_prereg_path: str | Path = DEFAULT_VARIANTS_PREREG_PATH,
    corpus: CachedCorpus | None = None,
) -> RecurrenceSpreadArtifact:
    """Run the recurrence_spread variant on the cached real corpus and assemble the sealed artifact."""

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    prereg = _read_variants_prereg(variants_prereg_path)
    sesoi_f1 = prereg["sesoi_f1"]

    corpus = corpus or load_cached_corpus(
        cache_root=cache_root, foa_root=foa_root, metadata_root=metadata_root
    )
    split = corpus.split
    features_by_clip = corpus.features_by_clip

    train_density = corpus.train_onset_density()
    n_test_clips = corpus.n_test_clips()
    n_test_onsets = corpus.n_test_onsets()
    n_test_frames = corpus.n_test_frames()
    if n_test_onsets == 0:
        raise RecurrenceSpreadRefusal("the cached test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # The noisy-TV channel is affine-matched to the real test content marginals (same recipe as the base
    # real run). This is a 2000-frame generated channel, not the corpus, so no corpus re-featurization.
    featurizer = FrozenFeaturizer()
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in split.test], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_SpreadSeedRun] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_runs.append(
            _run_seed_spread(
                seed,
                split,
                features_by_clip,
                noise_features,
                bed_config,
                train_density,
                operating_rate,
            )
        )
    measured_wall_ns = max(1, time.perf_counter_ns() - started)

    budget_points = _build_budget_points(seed_runs)
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
        "variants_prereg_canonical_sha256": prereg["canonical_sha256"],
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
    beats_random = bool(dominates and sign_flip.one_sided_significant and mean_delta_exceeds_sesoi)
    verdict = VERDICT_MECHANICS_OK if beats_random else VERDICT_NULL

    spreading_block = {
        "variant_id": VARIANT_ID,
        "operating_firing_fraction": round(float(operating_rate), 12),
        "refractory_frames": REFRACTORY_FRAMES,
        "per_seed": [{"seed": run.seed, **run.spread_report} for run in seed_runs],
    }
    fire_spread = _fire_spread_diagnostic(seed_runs)

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
                "one real variant run is a mechanics demonstration; scientific confirmation needs the "
                "independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified, and the four-variant family cannot clear family-wise significance at n=5"
            ),
        },
    )

    body: dict[str, Any] = {
        "schema": RECURRENCE_SPREAD_SCHEMA,
        "variant_id": VARIANT_ID,
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
        "spreading": spreading_block,
        "fire_spread_diagnostic": fire_spread,
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
            "n_params": 0,
            "parameter_digest": corpus.featurizer_digest,
            "flops_per_frame": FLOPS_PER_FRAME,
        },
        "gate": {
            "variant_id": VARIANT_ID,
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "n_spread_params": N_SPREAD_PARAMS,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE_SPREAD,
            "signal_train_flops": seed_runs[0].signal_train_flops,
            "search_flops": seed_runs[0].search_flops,
            "total_train_flops": seed_runs[0].total_train_flops,
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing,
        },
        "real_corpus": {
            "producer_schema": RECURRENCE_SPREAD_SCHEMA,
            "cache_key": corpus.cache_key,
            "foa_root": str(corpus.foa_root),
            "metadata_root": str(corpus.metadata_root),
            "n_clips": len(corpus.clips),
            "split_rooms": dict(split.detail),
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": n_test_frames,
            "train_onset_density": round(float(train_density), 12),
            "operating_firing_fraction": round(float(operating_rate), 12),
        },
        "variants_prereg": {
            "path": prereg["path"],
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_f1": sesoi_f1,
            "provisional": False,
            "min_one_sided_p": prereg["min_one_sided_p"],
        },
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)

    return RecurrenceSpreadArtifact(
        artifact=body,
        verdict=verdict,
        detail={
            "dominates": dominates,
            "beats_random": beats_random,
            "mean_delta": float(sign_flip.mean_delta),
            "one_sided_p": float(sign_flip.one_sided_p),
            "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
            "sesoi_f1": sesoi_f1,
            "noisy_tv_at_chance": noisy_tv_at_chance,
            "measured_wall_ns": measured_wall_ns,
            "per_seed_deltas": [float(v) for v in deltas],
            "fire_spread_diagnostic": fire_spread,
            "spreading": spreading_block,
        },
    )


def write_recurrence_spread_artifact(
    artifact: dict[str, Any], out_path: str | Path = DEFAULT_ARTIFACT_PATH
) -> Path:
    """Write the sealed variant artifact as canonical JSON bytes so its on-disk digest is reproducible."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact))
    return path
