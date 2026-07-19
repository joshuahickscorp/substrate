"""Adversarial reproduction 1 (data-split axis): the swapped-fold real-data producer.

This is a net-new, additive component. It runs the sealed STARSS23 concurrent-source-counting bed end to
end on the REAL STARSS23 FOA subset with ONE thing varied: the room-fold partition is SWAPPED. The gate
trains and tunes on the rooms the sealed bed scored (fold-4) and is scored on the rooms the sealed bed
trained and tuned on (fold-3). Fold-3 and fold-4 share no room, so the swapped split is still genuinely
room-disjoint and clip-disjoint.

Everything else is held byte-identical to the sealed bed by IMPORTING it, never re-implementing it: the
frozen count featurizer and estimator, the trained count gate, the sealed coasted-count-MAE referee, the
matched-budget FLOP harness, the four controls (rate-matched-random primary, always-on, never-update,
noisy-TV), and the exact sign-flip statistic all come straight from ``count_producer`` and its sealed
dependencies. The private per-seed run, featurization, estimation, noisy-TV, and budget-point builders of
``count_producer`` are reused directly, so the ONLY difference from the sealed run is which rooms train and
which rooms score, plus the disjoint seed family (10..14) that also breaks the original's seed luck.

The SESOI is preregistered by the reused cost-benefit rule on the fold-3 test labels before any test score
is read (see ``count_repro_data_split_prereg``); this producer writes the sealed prereg first and records
its digest in the artifact. The verdict is a mechanics demonstration only: ``activation_allowed``,
``scientific_promotion``, and ``independent_scientific_confirmation`` are hardcoded false, and a single
reproduction can never be scientifically confirmed.

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
from mop.science.statistics import BOUNDED_CLAIM_VERB, exact_sign_flip, sesoi_check
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter
from .controls import at_chance
from .count_estimator import FLOPS_PER_REESTIMATE, FrozenCountEstimator
from .count_featurizer import D_CFEAT, FLOPS_PER_FRAME_COUNT, FrozenCountFeaturizer
from .count_gate import FLOPS_PER_INFERENCE, CountOnlineState
from .count_harness import (
    ARM_ALWAYS_ON,
    ARM_CANDIDATE,
    ARM_NEVER_UPDATE,
    ARM_RATE_MATCHED_RANDOM,
    COUNT_BED_ID,
    run_matched_budget,
)
from .count_labels import build_count_clips, change_density, coast_from_zero_mae
from .count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
    STAGE,
    STAGE3_REQUIREMENT_ID,
    CountProducerRefusal,
    RealCountBedConfig,
    _build_budget_points,
    _estimate_all,
    _featurize_all,
    _real_noisy_tv_features,
    _run_seed_real,
)
from .count_referee import COLD_START
from .count_repro_data_split_prereg import (
    DEFAULT_REPRO_PREREG_PATH,
    REPRO_AXIS,
    build_data_split_prereg,
    write_data_split_prereg,
)
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
    """Return the full-scale swapped-fold configuration with disjoint seeds 10 through 14."""

    return RealCountBedConfig(seeds=DATA_SPLIT_SEEDS)


# ---------------------------------------------------------------------------
# The one varied axis: the SWAPPED fold split. Train on the original test rooms, score on the original
# train rooms. Reuses only the adapter's native fold room ids; everything downstream is the sealed path.
# ---------------------------------------------------------------------------


def _swapped_fold_split(
    adapter: RealStarssAdapter, n_val_rooms: int
) -> tuple[tuple[Clip, ...], tuple[Clip, ...], tuple[Clip, ...], dict[str, Any]]:
    """Build train / val / test with the fold roles swapped: test is exactly the native fold-3 dev-train.

    The sealed bed uses test = fold-4 dev-test, val = last N fold-3 rooms, train = rest of fold-3. This
    reproduction swaps the two folds: train comes from fold-4 (the original test rooms), val is the last N
    fold-4 rooms by sorted id, and the score partition is the whole native fold-3 dev-train. Fold-3 and
    fold-4 are physically room-disjoint, so train, val, and test remain room-disjoint and clip-disjoint.
    """

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


@dataclass(frozen=True, slots=True)
class DataSplitReproArtifact:
    """The assembled sealed data-split reproduction artifact plus the mechanics-only receipt and detail."""

    artifact: dict[str, Any]
    prereg: dict[str, Any]
    verdict: str
    detail: dict[str, Any]

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def build_data_split_repro_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealCountBedConfig | None = None,
    prereg_path: str | Path = DEFAULT_REPRO_PREREG_PATH,
) -> DataSplitReproArtifact:
    """Run the whole counting bed on the real subset with the fold split swapped, and seal the artifact.

    The preregistration is written to disk before any test score is computed. ``timestamp`` is passed by
    the caller and never read from the wall clock inside a sealed body. Every scored path except the split
    is imported from the sealed bed, so the ONLY manipulated variable is which rooms train and which score.
    """

    config = config or default_data_split_config()
    featurizer = FrozenCountFeaturizer()
    estimator = FrozenCountEstimator()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    count_clips = build_count_clips(adapter, metadata_root)
    gt_by_clip = {cid: cc.count_track for cid, cc in count_clips.items()}

    features_by_clip = _featurize_all(adapter, featurizer)
    estimator_by_clip = _estimate_all(adapter, estimator)
    # THE ONE VARIED AXIS: the swapped fold split.
    train_clips, val_clips, test_clips, split_detail = _swapped_fold_split(adapter, config.n_val_rooms)

    # Structural facts used by the SESOI cost-benefit and the operating-point rule. All label-only or
    # constant; no test score is read to build the prereg.
    train_count_clips = [count_clips[c.clip_id] for c in train_clips]
    test_count_clips = [count_clips[c.clip_id] for c in test_clips]
    train_density = change_density(train_count_clips)
    n_test_clips = len(test_clips)
    n_test_frames = int(sum(clip.n_frames for clip in test_clips))
    n_test_changes = int(sum(cc.n_changes for cc in test_count_clips))
    test_coast_from_zero = coast_from_zero_mae(test_count_clips)
    if n_test_changes == 0:
        raise CountProducerRefusal("the swapped-fold test split carries no count changes to track")
    operating_rate = min(config.target_rates, key=lambda r: abs(r - train_density))

    # 1. Preregister the SESOI and analysis plan BEFORE reading any test-split score.
    prereg = build_data_split_prereg(
        timestamp=timestamp,
        operating_reestimate_fraction=operating_rate,
        n_test_clips=n_test_clips,
        n_test_changes=n_test_changes,
        n_test_frames=n_test_frames,
        train_change_density=train_density,
        coast_from_zero_mae=test_coast_from_zero,
    )
    prereg_written = write_data_split_prereg(prereg, prereg_path)
    sesoi_mae = float(prereg["sesoi"]["sesoi_mae"])

    # 2. Now run the paired seeds and score the swapped test split. The per-seed run, noisy-TV channel, and
    # budget-point builder are the sealed bed's own functions, reused unchanged.
    pooled_test_features = np.concatenate([features_by_clip[c.clip_id] for c in test_clips], axis=0)
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs = []
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
    # Sign-flip statistic: delta_i = MAE_rate_matched_random(i) - MAE_candidate(i). Positive = candidate
    # reduces error. The exact_sign_flip test is one-sided upper tail on these deltas.
    deltas = [
        block["arm_scores"][PRIMARY_CONTROL]["mae"] - block["arm_scores"][ARM_CANDIDATE]["mae"]
        for block in per_seed
    ]
    sign_flip = exact_sign_flip(deltas)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1=sesoi_mae, provisional=False)
    mean_delta_exceeds_sesoi = bool(sesoi.exceeds_sesoi)
    # Report-facing convention (task): candidate minus random, negative = candidate better (lower MAE).
    mean_delta_candidate_minus_random = -float(sign_flip.mean_delta)

    stats_block = {
        "metric": "coasted-count-MAE",
        "delta_definition": (
            "delta_i = MAE_rate_matched_random(i) - MAE_candidate(i); positive = candidate lower error"
        ),
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

    # Shared per-clip tracks, sealed once (identical across seeds): the ground-truth count track and the
    # frozen estimator track the verifier re-coasts every arm against.
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
            "question": "concurrent-source counting under a swapped room-fold partition",
            "candidate_strictly_dominates_rate_matched_random": dominates,
            "one_sided_p": float(sign_flip.one_sided_p),
            "note": (
                "one swapped-fold reproduction is a mechanics demonstration; scientific confirmation needs "
                "the independent verifier plus at least three bias-independent reproductions and cannot be "
                "self-certified"
            ),
        },
    )

    truncations = [t.payload() for t in adapter.truncations()]
    dropped_onsets = sum(t["dropped_onsets_past_end"] for t in truncations)
    capped_clips = sum(1 for t in truncations if t["capped_by_max_frames"])

    body: dict[str, Any] = {
        "schema": REPRO_ARTIFACT_SCHEMA,
        "reproduction_axis": REPRO_AXIS,
        "of_bed": COUNT_BED_ID,
        "stage": STAGE,
        "bed_id": COUNT_BED_ID,
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
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": CountOnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "downstream_flops_per_reestimate": config.downstream_flops_per_reestimate,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // config.downstream_flops_per_reestimate,
        },
        "real_corpus": {
            "producer_schema": REPRO_PRODUCER_SCHEMA,
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
            "canonical_sha256": prereg["canonical_sha256"],
            "sesoi_mae": sesoi_mae,
            "provisional": False,
            "written_before_test_scores": True,
        },
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)

    return DataSplitReproArtifact(
        artifact=body,
        prereg=prereg,
        verdict=verdict,
        detail={
            "dominates": dominates,
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


DEFAULT_REPRO_ARTIFACT_PATH = Path("proof/STARSS23_COUNTING_REPRO_data_split.json")


def write_data_split_artifact(
    artifact: dict[str, Any], out_path: str | Path = DEFAULT_REPRO_ARTIFACT_PATH
) -> Path:
    """Write the sealed artifact as canonical JSON bytes so its on-disk digest is reproducible."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact))
    return path
