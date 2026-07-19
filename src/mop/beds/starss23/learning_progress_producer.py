
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    FlopModel,
    arm_flop_model,
)
from mop.science.statistics import (
    BOUNDED_CLAIM_VERB,
    FORBIDDEN_CLAIM_VERBS,
)
from mop.substrate.events import canonical_sha256, write_canonical_json

from . import BED_ID, CLAIM_SCOPE
from .adapter import onset_density as _onset_density
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
from .feature_cache import DEFAULT_CACHE_ROOT, load_cached_corpus
from .featurizer import FLOPS_PER_FRAME, FrozenFeaturizer
from .featurizer_variant_producer import (
    FeaturizerVariantSpec,
    VariantContext,
    VariantCorpus,
    build_featurizer_variant_artifact,
)
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
)
from .referee import COLLAR_MS, COLLAR_RULE, MATCH_RULE, PR_RULE, score_arm
from .schema import COLLAR_FRAMES, Clip, ClipSplit

LP_PRODUCER_SCHEMA = "mop-starss23-escs-learning-progress-producer/v1"
VARIANT_ID = "learning_progress"

DEFAULT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_learning_progress.json")
DEFAULT_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_learning_progress.prereg.json")
DEFAULT_VARIANTS_PREREG_PATH = Path("proof/STARSS23_ESCS_BED_VARIANTS.prereg.json")


class LPProducerRefusal(ValueError):
    pass




def _read_variants_prereg_digest(path: Path = DEFAULT_VARIANTS_PREREG_PATH) -> str | None:

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




def _train_lp_gate(
    seed: int,
    split_train: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[LearningProgressGate, int]:

    train_features = np.concatenate(
        [features_by_clip[clip.clip_id] for clip in split_train], axis=0
    )
    gate = LearningProgressGate(seed=seed)
    gate.fit(train_features, epochs=epochs, learning_rate=learning_rate)
    return gate, int(train_features.shape[0])


def _adjacency_fraction(fires: list[int]) -> tuple[int, int]:

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



def _flop_model_lp(
    kind: str,
    total_frames: int,
    train_frames: int,
    gate: LearningProgressGate,
    epochs: int,
) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME,
        gate_infer_per_frame=gate.flops_per_inference(),
        downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        candidate_train_flops=lambda: gate.training_flops(train_frames, epochs),
    )




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

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    corpus = load_cached_corpus(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=config.max_frames,
        n_val_rooms=config.n_val_rooms,
    )
    prepared = VariantCorpus(
        split=corpus.split,
        features_by_clip=corpus.features_by_clip,
        train_density=_onset_density(corpus.split.train),
        n_test_clips=len(corpus.split.test),
        n_test_onsets=sum(len(clip.onsets) for clip in corpus.split.test),
        n_test_frames=int(sum(clip.n_frames for clip in corpus.split.test)),
    )
    featurizer = FrozenFeaturizer()
    reference_gate = LearningProgressGate(seed=config.seeds[0])
    completed_runs: list[_LPSeedRun] = []
    prereg_holder: dict[str, Any] = {}

    def prepare_prereg(current: VariantCorpus) -> tuple[dict[str, Any], str | Path]:
        prereg = build_lp_prereg(
            timestamp=timestamp,
            n_test_clips=current.n_test_clips,
            n_test_onsets=current.n_test_onsets,
            n_test_frames=current.n_test_frames,
            train_onset_density=current.train_density,
            operating_firing_fraction=min(
                config.target_rates,
                key=lambda rate: abs(rate - current.train_density),
            ),
            n_seeds=len(config.seeds),
        )
        prereg_holder.update(prereg)
        return prereg, write_canonical_json(prereg, prereg_path)

    def run_seed(
        seed: int,
        current: VariantCorpus,
        noise_features: np.ndarray,
        _current_bed_config: Any,
    ) -> _LPSeedRun:
        run = _run_seed_lp(
            seed,
            current.split,
            current.features_by_clip,
            noise_features,
            config,
            current.train_density,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        completed_runs.append(run)
        return run

    def diagnostics_payload() -> dict[str, Any]:
        candidate_fires = sum(run.diagnostics["candidate_fires"] for run in completed_runs)
        candidate_adjacent = sum(
            run.diagnostics["candidate_adjacent_fires"] for run in completed_runs
        )
        candidate_tp = sum(
            run.diagnostics["candidate_distinct_onset_tp"] for run in completed_runs
        )
        random_tp = sum(
            run.diagnostics["rate_matched_random_distinct_onset_tp"]
            for run in completed_runs
        )
        return {
            "operating_point_rule": "swept budget closest to the train onset density",
            "pooled_over_seeds": {
                "candidate_fires": candidate_fires,
                "candidate_adjacent_fires": candidate_adjacent,
                "candidate_adjacency_fraction": (
                    round(candidate_adjacent / candidate_fires, 12)
                    if candidate_fires
                    else 0.0
                ),
                "candidate_distinct_onset_tp": candidate_tp,
                "rate_matched_random_distinct_onset_tp": random_tp,
                "committed_baseline_distinct_onset_tp": 204,
                "committed_random_distinct_onset_tp": 237,
            },
            "per_seed": [run.diagnostics for run in completed_runs],
        }

    def featurizer_payload(_context: VariantContext) -> dict[str, Any]:
        return {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        }

    def extra_payload(context: VariantContext) -> dict[str, Any]:
        return {
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "variant": {
                "variant_id": VARIANT_ID,
                "producer_schema": LP_PRODUCER_SCHEMA,
                "exploratory_fifth_variant": True,
                "hypothesis": prereg_holder["hypothesis"],
            },
            "fire_spread_diagnostic": context.spread,
            "referee": {
                "collar_frames": COLLAR_FRAMES,
                "collar_ms": COLLAR_MS,
                "collar_rule": COLLAR_RULE,
                "match_rule": MATCH_RULE,
                "pr_rule": PR_RULE,
            },
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_firing": DOWNSTREAM_FLOPS_PER_FIRING,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // DOWNSTREAM_FLOPS_PER_FIRING
                ),
            },
            "real_corpus": {
                "producer_schema": LP_PRODUCER_SCHEMA,
                "cache_key": corpus.cache_key,
                "cache_dir": str(corpus.cache_dir),
                "featurizer_digest": corpus.featurizer_digest,
                "foa_root": str(Path(foa_root)),
                "metadata_root": str(Path(metadata_root)),
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
                "path": str(Path(prereg_path)),
                "canonical_sha256": context.prereg_digest,
                "sesoi_f1": context.sesoi_f1,
                "provisional": False,
                "written_before_test_scores": True,
                "sealed_variants_prereg_canonical_sha256": prereg_holder[
                    "sealed_variants_prereg_canonical_sha256"
                ],
            },
        }

    spec = FeaturizerVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        variant_id=VARIANT_ID,
        identity_key="variant_id",
        prereg_schema="",
        prereg_family_field="",
        prereg_member_field="",
        refusal=LPProducerRefusal,
        flops_per_frame=FLOPS_PER_FRAME,
        spread=lambda _per_seed: diagnostics_payload(),
        featurizer_payload=featurizer_payload,
        extra_payload=extra_payload,
        final_extra=lambda context: {
            "pooled_adjacency_fraction": context.spread["pooled_over_seeds"][
                "candidate_adjacency_fraction"
            ],
            "pooled_candidate_distinct_onset_tp": context.spread["pooled_over_seeds"][
                "candidate_distinct_onset_tp"
            ],
            "pooled_rate_matched_random_distinct_onset_tp": context.spread[
                "pooled_over_seeds"
            ]["rate_matched_random_distinct_onset_tp"],
        },
        receipt_extra={},
        run_seed=run_seed,
        receipt_note=(
            "one real run of an exploratory gate variant is a mechanics demonstration; scientific "
            "confirmation needs the independent verifier plus at least three bias-independent "
            "reproductions and cannot be self-certified"
        ),
        prepare_prereg=prepare_prereg,
        include_prereg_in_result=True,
        flop_model=lambda kind, seed_runs, _current_bed_config: _flop_model_lp(
            kind,
            seed_runs[0].total_frames,
            seed_runs[0].train_frames,
            reference_gate,
            epochs,
        ),
        include_spread_in_detail=False,
        gate_payload=lambda _context: {
            "variant_id": VARIANT_ID,
            "params": completed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": reference_gate.state_bytes(),
            "flops_per_inference": reference_gate.flops_per_inference(),
            "parameter_digest": reference_gate.parameter_digest(),
        },
    )
    return build_featurizer_variant_artifact(
        config=config,
        bed_config=bed_config,
        corpus=prepared,
        featurizer=featurizer,
        prereg_path=prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )
