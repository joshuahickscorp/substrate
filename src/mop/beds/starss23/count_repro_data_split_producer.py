
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mop.science import ArtifactResult

from .adapter import RealStarssAdapter
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import FLOPS_PER_INFERENCE, CountOnlineState
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    CountProducerRefusal,
    RealCountBedConfig,
    _flop_model,
    _real_noisy_tv_features,
    _run_seed_real,
)
from .count_repro_data_split_prereg import (
    DEFAULT_REPRO_PREREG_PATH,
    REPRO_AXIS,
    build_data_split_prereg,
)
from .count_variant_producer import (
    CountVariantContext,
    CountVariantSpec,
    build_count_variant_artifact,
    prepare_count_variant_corpus,
)
from .experiments import COUNT_BED_ID
from .schema import Clip

REPRO_PRODUCER_SCHEMA = "mop-starss23-count-repro-data-split-producer/v1"
# A distinct artifact schema so the ORIGINAL sealed verifier rejects this file and only the separately
# authored data-split verifier accepts it. This is a reproduction of the same bed, not the sealed run.
REPRO_ARTIFACT_SCHEMA = "mop-starss23-escs-count-bed-repro-data-split/v1"

# Disjoint seed family so this reproduction shares none of the original's seed luck (the doc flags that
# original seed 3 carried most of the effect). derive_seed32 passes these in-range integers through
# unchanged, so they give genuinely independent gate inits AND independent rate-matched-random draws.
DATA_SPLIT_SEEDS: tuple[int, ...] = (10, 11, 12, 13, 14)


def default_data_split_config() -> RealCountBedConfig:

    return RealCountBedConfig(seeds=DATA_SPLIT_SEEDS)


# ---------------------------------------------------------------------------
# The one varied axis: the SWAPPED fold split. Train on the original test rooms, score on the original
# train rooms. Reuses only the adapter's native fold room ids; everything downstream is the sealed path.
# ---------------------------------------------------------------------------


def _swapped_fold_split(
    adapter: RealStarssAdapter, n_val_rooms: int
) -> tuple[tuple[Clip, ...], tuple[Clip, ...], tuple[Clip, ...], dict[str, Any]]:

    dev = adapter.dev_split()
    by_id = {clip.clip_id: clip for clip in adapter.clips()}
    fold3 = [by_id[cid] for cid in dev.dev_train]  # sealed-bed train+val source; here the SCORE partition
    fold4 = [by_id[cid] for cid in dev.dev_test]  # sealed-bed test source; here the TRAIN+VAL source
    fold4_rooms = sorted({clip.room_id for clip in fold4})
    if n_val_rooms <= 0 or n_val_rooms >= len(fold4_rooms):
        raise CountProducerRefusal(
            f"n_val_rooms must leave at least one train room; saw {n_val_rooms} of {len(fold4_rooms)} "
            "fold-4 rooms"
        )
    val_rooms = set(fold4_rooms[-n_val_rooms:])
    train = tuple(sorted((c for c in fold4 if c.room_id not in val_rooms), key=lambda c: c.clip_id))
    val = tuple(sorted((c for c in fold4 if c.room_id in val_rooms), key=lambda c: c.clip_id))
    test = tuple(sorted(fold3, key=lambda c: c.clip_id))
    if not train or not val or not test:
        raise CountProducerRefusal("the swapped-fold split produced an empty partition")
    train_rooms = {c.room_id for c in train}
    test_rooms = {c.room_id for c in test}
    if train_rooms & val_rooms or train_rooms & test_rooms or val_rooms & test_rooms:
        raise CountProducerRefusal("the swapped-fold split is not room-disjoint")
    detail = {
        "train_rooms": sorted(train_rooms),
        "val_rooms": sorted(val_rooms),
        "test_rooms": sorted(test_rooms),
        "split_rule": (
            "SWAP of the sealed bed: train = native fold-4 dev-test rooms minus the last N val rooms; "
            "val = last N fold-4 rooms by sorted id; test = the whole native fold-3 dev-train; room-disjoint"
        ),
        "swapped_from_sealed": True,
    }
    return train, val, test, detail


# ---------------------------------------------------------------------------
# Assemble and seal the swapped-fold reproduction artifact.
# ---------------------------------------------------------------------------


def build_data_split_repro_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_REPRO_PREREG_PATH,
) -> ArtifactResult:

    config = config or default_data_split_config()
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
        split_provider=lambda current_adapter: _swapped_fold_split(
            current_adapter, config.n_val_rooms
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
            "params": context.seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        }

    def estimator_payload(_context: CountVariantContext) -> dict[str, Any]:
        return {
            "n_params": estimator.n_params(),
            "parameter_digest": estimator.parameter_digest(),
            "flops_per_reestimate": FLOPS_PER_REESTIMATE,
        }

    spec = CountVariantSpec(
        artifact_schema=REPRO_ARTIFACT_SCHEMA,
        producer_schema=REPRO_PRODUCER_SCHEMA,
        refusal=CountProducerRefusal,
        no_changes_message="the swapped-fold test split carries no count changes to track",
        score_field="mae",
        build_prereg=lambda current: build_data_split_prereg(
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
        ),
        flop_model=lambda kind, total_frames, train_frames: _flop_model(
            kind, total_frames, train_frames, config
        ),
        featurizer_payload=featurizer_payload,
        gate_payload=gate_payload,
        estimator_payload=estimator_payload,
        receipt_detail=lambda _context: {
            "reproduction_axis": REPRO_AXIS,
            "question": "concurrent-source counting under a swapped room-fold partition",
            "note": (
                "one swapped-fold reproduction is a mechanics demonstration; scientific confirmation "
                "needs the independent verifier plus at least three bias-independent reproductions and "
                "cannot be self-certified"
            ),
        },
        artifact_extra=lambda _context: {
            "reproduction_axis": REPRO_AXIS,
            "of_bed": COUNT_BED_ID,
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
                "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
                "break_even_frames_anchor": (
                    FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate
                ),
            },
        },
        prereg_extra=lambda _context: {},
        final_extra=lambda _context: {},
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


DEFAULT_REPRO_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_data_split.json")
