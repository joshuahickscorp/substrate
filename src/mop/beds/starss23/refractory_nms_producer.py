
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
)
from mop.science.gating import assemble_causal_inputs, causal_gate_trace

from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
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
from .feature_cache import DEFAULT_CACHE_ROOT, CachedCorpus, load_cached_corpus
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .featurizer_variant_producer import (
    FeaturizerVariantSpec,
    VariantContext,
    VariantCorpus,
    build_featurizer_variant_artifact,
)
from .gate import CandidateGate, OnlineState
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
)
from .referee import fire_spread, score_arm, summarize_fire_spread
from .schema import COLLAR_FRAMES, ClipSplit

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-refractory-nms/v1"
VARIANT_ID = "refractory_nms"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_refractory_nms.json")


class RefractoryNmsRefusal(ValueError):
    pass


def _variant_hypothesis() -> str:
    for entry in GATE_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise RefractoryNmsRefusal(f"variant {VARIANT_ID!r} is not in the sealed variant family")


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

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in split_train:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, OnlineState.initial))
        targets.append(_voc_targets(clip.onset_frames, clip.n_frames, window=voc_window))
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    gate = RefractoryNmsGate(seed=seed, window=window)
    gate.fit(x, y, epochs=epochs, learning_rate=learning_rate, ponder_lambda=ponder_lambda)
    return gate, int(x.shape[0])


# ---------------------------------------------------------------------------
# Fire-spread diagnostics: adjacency and distinct-onset true positives.
# ---------------------------------------------------------------------------


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
            "candidate": fire_spread(variant_candidate),
            "rate_matched_random": fire_spread(variant_rmr),
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

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in split.train:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, OnlineState.initial))
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
        [causal_gate_trace(gate, features_by_clip[clip.clip_id], 0.5, OnlineState.initial)[1]
         for clip in split.val]
    )
    rates = {rate: float(np.quantile(val_probs, 1.0 - rate)) for rate in config.target_rates}
    operating_rate = min(rates, key=lambda r: abs(r - operating_density))
    theta = rates[operating_rate]

    candidate_pairs: list[tuple[list[int], list[int]]] = []
    rmr_pairs: list[tuple[list[int], list[int]]] = []
    for clip in split.test:
        gt = list(clip.onset_frames)
        candidate_fires, _ = causal_gate_trace(
            gate, features_by_clip[clip.clip_id], theta, OnlineState.initial
        )
        rmr_fires = rate_matched_random_fires(
            candidate_fires, clip.n_frames, seed=seed, clip_id=clip.clip_id
        )
        candidate_pairs.append((gt, candidate_fires))
        rmr_pairs.append((gt, rmr_fires))
    return {
        "operating_rate": operating_rate,
        "candidate": fire_spread(candidate_pairs),
        "rate_matched_random": fire_spread(rmr_pairs),
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

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    corpus = corpus or load_cached_corpus(cache_root=cache_root)
    prepared = VariantCorpus(
        split=corpus.split,
        features_by_clip=corpus.features_by_clip,
        train_density=corpus.train_onset_density(),
        n_test_clips=corpus.n_test_clips(),
        n_test_onsets=corpus.n_test_onsets(),
        n_test_frames=corpus.n_test_frames(),
    )
    featurizer = FrozenFeaturizer()
    diagnostics: list[dict[str, Any]] = []

    def run_seed(
        seed: int,
        current: VariantCorpus,
        noise_features: np.ndarray,
        current_bed_config: Any,
    ) -> _SeedRun:
        seed_run, diagnostic = _run_seed_variant(
            seed,
            current.split,
            current.features_by_clip,
            noise_features,
            current_bed_config,
            current.train_density,
            window,
        )
        diagnostics.append(diagnostic)
        return seed_run

    def featurizer_payload(_context: VariantContext) -> dict[str, Any]:
        return {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
            "feature_cache_key": corpus.cache_key,
            "note": (
                "featurized once and cached; the FLOP ledger charges it per arm from the cache count"
            ),
        }

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "variant_id": VARIANT_ID,
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": context.beats_random,
            "variant": {
                "variant_id": VARIANT_ID,
                "hypothesis": _variant_hypothesis(),
                "firing_policy": (
                    "collar-width refractory non-maximum suppression on the p_fire trace"
                ),
                "window_frames": int(window),
                "window_ms": int(window) * 100,
                "only_firing_policy_differs": True,
                "trained_weights_identical_to_committed_gate": True,
                "variants_prereg_path": str(Path(variants_prereg_path)),
                "variants_prereg_canonical_sha256": context.prereg_digest,
                "fire_spread_diagnostic": context.spread,
            },
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing
                ),
            },
            "real_corpus": {
                "producer_schema": REAL_PRODUCER_SCHEMA,
                "variant_producer_schema": VARIANT_ARTIFACT_SCHEMA,
                "foa_root": str(Path(DEFAULT_FOA_ROOT)),
                "metadata_root": str(Path(DEFAULT_METADATA_ROOT)),
                "feature_cache_key": corpus.cache_key,
                "n_clips": len(corpus.clips),
                "split_rooms": dict(corpus.split.detail),
                "n_train_frames": context.seed_runs[0].train_frames,
                "n_test_clips": prepared.n_test_clips,
                "n_test_onsets": prepared.n_test_onsets,
                "n_test_frames": prepared.n_test_frames,
                "train_onset_density": round(float(prepared.train_density), 12),
                "operating_firing_fraction": round(float(context.operating_rate), 12),
            },
            "prereg": {
                "path": str(Path(variants_prereg_path)),
                "canonical_sha256": context.prereg_digest,
                "sesoi_f1": context.sesoi_f1,
                "provisional": False,
                "written_before_test_scores": True,
                "rebuilt_by_this_producer": False,
            },
        }

    spec = FeaturizerVariantSpec(
        artifact_schema=VARIANT_ARTIFACT_SCHEMA,
        variant_id=VARIANT_ID,
        identity_key="variant_id",
        prereg_schema=VARIANTS_PREREG_SCHEMA,
        prereg_family_field="variants",
        prereg_member_field="variant_id",
        refusal=RefractoryNmsRefusal,
        flops_per_frame=FLOPS_PER_FRAME,
        spread=lambda _per_seed: _assemble_spread_diagnostic(diagnostics),
        featurizer_payload=featurizer_payload,
        extra_payload=extra_payload,
        final_extra=lambda _context: {},
        receipt_extra={},
        run_seed=run_seed,
        prereg_family_label="variant",
        receipt_note=(
            "one real run of one gate variant is a mechanics outcome; scientific confirmation needs "
            "the independent verifier plus at least three bias-independent reproductions and, for this "
            "four-variant family at n equals 5, cannot clear family-wise significance at all"
        ),
    )
    return build_featurizer_variant_artifact(
        config=config,
        bed_config=bed_config,
        corpus=prepared,
        featurizer=featurizer,
        prereg_path=variants_prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )


def _assemble_spread_diagnostic(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:

    def _summary(path: tuple[str, ...]) -> dict[str, Any]:
        per_seed = []
        for diag in diagnostics:
            node = diag
            for key in path:
                node = node[key]
            per_seed.append(node)
        return summarize_fire_spread(per_seed)

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
