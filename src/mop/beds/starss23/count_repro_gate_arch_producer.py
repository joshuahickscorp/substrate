"""Two-hidden-layer gate reproduction producer for the STARSS23 concurrent-count bed.

Provider, split, preregistration, gate, estimator, FLOP, and evidence declarations remain local; the
shared count-variant authority owns only the held-fixed scoring and sealed-artifact lifecycle."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mop.science import ArtifactResult
from mop.science.budget import (
    ARM_RATE_MATCHED_RANDOM,
    FlopModel,
    arm_flop_model,
)
from mop.science.gating import assemble_causal_inputs

from .adapter import RealStarssAdapter, native_fold_split
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import CountOnlineState

# Gate-agnostic pipeline stages reused by reference from the sealed producer so they cannot drift.
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    CountProducerRefusal,
    RealCountBedConfig,
    _real_noisy_tv_features,
    _run_seed_real,
)
from .count_repro_gate_arch_gate import (
    D_IN_GATE_ARCH,
    FLOPS_PER_INFERENCE_GATE_ARCH,
    HIDDEN1,
    HIDDEN2,
    N_OUT,
    REPRO_AXIS,
    CountReproGateArchGate,
    training_flops_two_layer,
    voc_targets_from_count_track,
)
from .count_repro_gate_arch_prereg import (
    COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
    DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
    build_count_repro_gate_arch_prereg,
)
from .count_variant_producer import (
    CountVariantContext,
    CountVariantSpec,
    build_count_variant_artifact,
    prepare_count_variant_corpus,
)
from .schema import Clip

COUNT_REPRO_GATE_ARCH_PRODUCER_SCHEMA = "mop-starss23-count-repro-gate-arch-producer/v1"
ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-gate-arch/v1"
STAGE = 3
PRIMARY_CONTROL = ARM_RATE_MATCHED_RANDOM
STAGE3_REQUIREMENT_ID = "stage3.confirmed_useful_mechanism"

# The disjoint gate-architecture seed family: no overlap with the original (0..4), so no shared seed luck.
GATE_ARCH_SEEDS: tuple[int, ...] = (40, 41, 42, 43, 44)


class CountReproGateArchProducerRefusal(ValueError):
    """Raised when the gate-architecture reproduction cannot assemble a well-formed sealed artifact."""


def _train_count_gate(
    seed: int,
    train_clips: tuple[Clip, ...],
    features_by_clip: dict[str, np.ndarray],
    gt_by_clip: dict[str, tuple[int, ...]],
    config: RealCountBedConfig,
) -> tuple[CountReproGateArchGate, int]:
    """Train the re-authored two-layer gate on train-room value-of-computation targets.

    The shared causal input assembly and ``voc_targets_from_count_track`` are held fixed; only the gate
    class is the varied axis.
    """

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for clip in train_clips:
        features = features_by_clip[clip.clip_id]
        inputs.append(assemble_causal_inputs(features, CountOnlineState.initial))
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


def _flop_model(
    kind: str, total_frames: int, train_frames: int, config: RealCountBedConfig
) -> FlopModel:
    return arm_flop_model(
        kind,
        total_frames,
        featurize_per_frame=FLOPS_PER_FRAME_COUNT,
        gate_infer_per_frame=FLOPS_PER_INFERENCE_GATE_ARCH,
        downstream_flops_per_firing=config.downstream_flops_per_reestimate,
        candidate_train_flops=lambda: training_flops_two_layer(train_frames, config.epochs),
    )


# ---------------------------------------------------------------------------
# Assemble and seal the gate-architecture reproduction artifact.
# ---------------------------------------------------------------------------


def build_real_count_repro_gate_arch_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_COUNT_REPRO_GATE_ARCH_PREREG_PATH,
) -> ArtifactResult:
    """Run the two-hidden-layer gate declaration through the shared count lifecycle."""

    config = config or RealCountBedConfig(seeds=GATE_ARCH_SEEDS)
    featurizer = FrozenCountFeaturizer()
    estimator = FrozenCountEstimator()
    adapter = RealStarssAdapter(
        foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames
    )
    corpus = prepare_count_variant_corpus(
        adapter=adapter,
        foa_root=foa_root,
        metadata_root=metadata_root,
        featurizer=featurizer,
        estimator=estimator,
        config=config,
        split_provider=lambda current_adapter: native_fold_split(
            current_adapter,
            config.n_val_rooms,
            refusal=CountProducerRefusal,
        ),
    )

    def featurizer_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME_COUNT,
            "d_cfeat": D_CFEAT,
        }

    def gate_payload(context: CountVariantContext) -> dict[str, Any]:
        return {
            "architecture": "two_hidden_layer_mlp",
            "topology": "264 -> 8 -> 4 -> 1",
            "d_in": D_IN_GATE_ARCH,
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "n_out": N_OUT,
            "params": context.seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "parameter_digest_seed0": CountReproGateArchGate(
                seed=config.seeds[0]
            ).parameter_digest(),
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE_GATE_ARCH,
            "sealed_gate_topology": "264 -> 12 -> 1",
            "sealed_gate_flops_per_inference": 6385,
        }

    def estimator_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        }

    def operating_point(context: CountVariantContext) -> tuple[str, dict[str, Any], bool]:
        budget_id = context.seed_runs[0].operating_budget_id
        row = next(
            item
            for item in context.report.per_budget_candidate_vs_rate_matched_random
            if item["budget_id"] == budget_id
        )
        survives = bool(
            row["candidate_strictly_beats_rate_matched_random"]
            and context.mean_delta_exceeds_sesoi
            and context.sign_flip.one_sided_significant
        )
        return budget_id, row, survives

    def artifact_extra(context: CountVariantContext) -> dict[str, Any]:
        budget_id, row, survives = operating_point(context)
        return {
            "reproduction_axis": REPRO_AXIS,
            "survives_operating_point": survives,
            "operating_point": {
                "budget_id": budget_id,
                "candidate_mean_mae": float(row["candidate_mean_mae"]),
                "rate_matched_random_mean_mae": float(
                    row["rate_matched_random_mean_mae"]
                ),
                "candidate_strictly_beats_rate_matched_random": bool(
                    row["candidate_strictly_beats_rate_matched_random"]
                ),
            },
        }

    def final_extra(context: CountVariantContext) -> dict[str, Any]:
        budget_id, row, survives = operating_point(context)
        return {
            "survives_operating_point": survives,
            "operating_budget_id": budget_id,
            "candidate_operating_mae": float(row["candidate_mean_mae"]),
            "rate_matched_random_operating_mae": float(
                row["rate_matched_random_mean_mae"]
            ),
        }

    spec = CountVariantSpec(
        artifact_schema=ARTIFACT_SCHEMA,
        producer_schema=COUNT_REPRO_GATE_ARCH_PRODUCER_SCHEMA,
        refusal=CountReproGateArchProducerRefusal,
        no_changes_message="the real test split carries no count changes to track",
        score_field="mae",
        build_prereg=lambda current: build_count_repro_gate_arch_prereg(
            timestamp=timestamp,
            operating_reestimate_fraction=current.operating_rate,
            n_test_clips=current.n_test_clips,
            n_test_changes=current.n_test_changes,
            n_test_frames=current.n_test_frames,
            train_change_density=current.train_density,
            coast_from_zero_mae=current.test_coast_from_zero,
        ),
        noise_features=_real_noisy_tv_features,
        run_seed=lambda seed, current, noise: _run_seed_real(
            seed,
            current.train_clips,
            current.val_clips,
            current.test_clips,
            current.features_by_clip,
            current.estimator_by_clip,
            current.gt_by_clip,
            noise,
            config,
            current.train_density,
            train_gate_provider=_train_count_gate,
        ),
        flop_model=lambda kind, total_frames, train_frames: _flop_model(
            kind, total_frames, train_frames, config
        ),
        featurizer_payload=featurizer_payload,
        gate_payload=gate_payload,
        estimator_payload=estimator_payload,
        receipt_detail=lambda _context: {
            "reproduction_axis": REPRO_AXIS,
            "question": (
                "gate-architecture reproduction of concurrent-source counting: does a two-hidden-layer "
                "gate beat rate-matched-random at matched budget"
            ),
            "note": (
                "one real reproduction is a mechanics demonstration; scientific confirmation needs the "
                "independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified"
            ),
        },
        artifact_extra=artifact_extra,
        prereg_extra=lambda _context: {
            "schema": COUNT_REPRO_GATE_ARCH_PREREG_SCHEMA,
        },
        final_extra=final_extra,
    )
    return build_count_variant_artifact(
        config=config,
        corpus=corpus,
        featurizer=featurizer,
        estimator=estimator,
        prereg_path=prereg_path,
        spec=spec,
        clock_ns=time.perf_counter_ns,
    )


DEFAULT_COUNT_REPRO_GATE_ARCH_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_gate_arch.json")
