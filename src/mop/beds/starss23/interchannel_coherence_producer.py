"""Real-data producer for the F1 featurizer "interchannel_coherence" on the STARSS23 FOA subset.

This is a net-new, additive component. It runs the frozen interchannel-coherence featurizer end to end on
the REAL, MIT-licensed STARSS23 FOA subset and assembles a sealed
``proof/STARSS23_ESCS_BED_interchannel_coherence.json`` with the same evidence shape as the committed
``proof/STARSS23_ESCS_BED.json``. It changes NONE of the sealed scoring logic: the referee, the exact
sign-flip statistics, the matched-budget harness, and every control are imported unchanged, and the
value-of-computation training-target assembly, the causal p_fire pass, the gate training, and the
fold-respecting split are imported unchanged from the committed real producer so the featurizer-swap lane
scores through the identical machinery. The ONLY thing that differs from the committed real run is the
FROZEN front-end: ``InterchannelCoherenceFeaturizer`` replaces ``FrozenFeaturizer`` and its per-frame
FLOPs are charged into a purpose-built FLOP model so every arm charges the new featurizer's honest cost.

The new featurizer emits a different 256-vector than the frozen cache stores (a different parameter
digest, so a different cache key), so it is featurized INLINE here, exactly as the committed real producer
featurizes the frozen front-end inline. The 45-clip subset featurizes in a few seconds.

The SESOI and the whole analysis plan are read from the already-sealed featurizer preregistration
``proof/STARSS23_ESCS_BED_interchannel_coherence.prereg.json``; this producer NEVER rebuilds or reweakens
it. The verdict is a mechanics outcome only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single run at n equals 5 across the
three-featurizer family can never promote.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from mop.ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
)
from mop.science import (
    ArtifactResult,
    artifact_envelope,
    demonstration_receipt,
    finalize_artifact,
    read_sealed_prereg_member,
    safety_flags,
)
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
from mop.science.statistics import exact_sign_flip, sign_flip_payload

from . import BED_ID, FLOP_CEILING, STAGE3_FORCING_NULL
from .adapter import RealStarssAdapter, map_clip_audio, native_fold_split
from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_FEATURIZE,
    PRIMARY_CONTROL,
)
from .controls import at_chance
from .experiments import ONSET_BUDGET_POLICY
from .featurizer_interchannel_coherence import (
    D_FEAT,
    FLOPS_PER_FRAME,
    InterchannelCoherenceFeaturizer,
)
from .gate import FLOPS_PER_INFERENCE, OnlineState, training_flops
from .interchannel_coherence_prereg import (
    DEFAULT_FEATURIZERS_PREREG_PATH,
    FEATURIZER_VARIANTS,
    FEATURIZERS_PREREG_SCHEMA,
)
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealArtifactRefusal,
    RealBedConfig,
    _onset_density,
    _real_noisy_tv_features,
    _run_seed_real,
)
from .referee import summarize_fire_spread_blocks
from .schema import COLLAR_FRAMES

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-interchannel-coherence/v1"
VARIANT_ID = "interchannel_coherence"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_interchannel_coherence.json")


class InterchannelCoherenceRefusal(ValueError):
    """Raised when the interchannel_coherence producer cannot assemble a well-formed sealed artifact."""


def _variant_hypothesis() -> str:
    for entry in FEATURIZER_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise InterchannelCoherenceRefusal(f"featurizer {VARIANT_ID!r} is not in the sealed featurizer family")


# ---------------------------------------------------------------------------
# FLOP model: the NEW featurizer's per-frame cost charged identically to every arm.
# ---------------------------------------------------------------------------


def _flop_model(kind: str, total_frames: int, train_frames: int, config: Any) -> FlopModel:
    """Full-lifecycle FLOP model charging the interchannel-coherence featurizer per-frame cost.

    This is the ONE place that differs in accounting from the committed producer: ``featurize`` charges
    ``FLOPS_PER_FRAME`` of the NEW featurizer (not the frozen log-mel front-end), so the matched-budget
    ledger prices the front-end actually run. The charge is applied identically to all four arms, so the
    candidate and the rate-matched-random control stay byte-equal in inference FLOPs and any accuracy gap
    remains attributable to WHERE compute is spent, not how much.
    """

    featurize = FLOPS_PER_FRAME * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = FLOPS_PER_INFERENCE * total_frames if runs_gate else 0
    train = training_flops(train_frames, config.epochs) if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=config.downstream_flops_per_firing,
        train_flops=train,
    )
# ---------------------------------------------------------------------------
# Fire-spread diagnostics: adjacency and distinct-onset true positives at the operating point.
# ---------------------------------------------------------------------------


def _assemble_spread_diagnostic(seed_runs: list[Any]) -> dict[str, Any]:
    """Summarize the per-seed fire-spread diagnostics for the candidate and the rate-matched-random arm."""

    per_seed = [run.per_seed_block for run in seed_runs]

    return {
        "definition": (
            "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of another "
            "fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one referee true "
            "positives at the operating budget."
        ),
        "collar_frames": COLLAR_FRAMES,
        "candidate": summarize_fire_spread_blocks(per_seed, ARM_CANDIDATE),
        "rate_matched_random": summarize_fire_spread_blocks(per_seed, ARM_RATE_MATCHED_RANDOM),
        "committed_null_seed0_anchor": {
            "candidate_distinct_onset_tp": 204,
            "rate_matched_random_distinct_onset_tp": 237,
            "candidate_adjacency_fraction_approx": 0.42,
            "source": "docs/mixture_of_perspectives/26_escs_starss23_bed.md",
        },
    }


# ---------------------------------------------------------------------------
# Assemble and seal the featurizer-swap artifact.
# ---------------------------------------------------------------------------


def build_interchannel_coherence_artifact(
    *,
    timestamp: str,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> ArtifactResult:
    """Run the interchannel_coherence featurizer on the real subset and assemble the sealed artifact.

    The SESOI and the analysis plan come from the already-sealed featurizer preregistration; this producer
    never rebuilds or reweakens it. ``timestamp`` is passed by the caller and never read from the wall
    clock inside a sealed body. The new featurizer is featurized inline and its per-frame FLOPs are charged
    to every arm.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    featurizer = InterchannelCoherenceFeaturizer()

    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=config.max_frames)
    features_by_clip = map_clip_audio(adapter, featurizer.featurize)
    split = native_fold_split(
        adapter, config.n_val_rooms, refusal=RealArtifactRefusal, refuse_empty=False
    )

    prereg = read_sealed_prereg_member(
        featurizers_prereg_path,
        expected_schema=FEATURIZERS_PREREG_SCHEMA,
        family_field="variants",
        member_field="variant_id",
        member_id=VARIANT_ID,
        family_label="featurizer",
        refusal=InterchannelCoherenceRefusal,
    )
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])
    prereg_digest = str(prereg["canonical_sha256"])

    # Structural facts (label-only), used for provenance and the operating-point rule.
    train_density = _onset_density(split.train)
    n_test_clips = len(split.test)
    n_test_onsets = sum(len(clip.onsets) for clip in split.test)
    n_test_frames = int(sum(clip.n_frames for clip in split.test))
    if n_test_onsets == 0:
        raise InterchannelCoherenceRefusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # noisy-TV marginals: match the injected white-noise channel to the real test feature marginals.
    pooled_test_features = np.concatenate(
        [features_by_clip[clip.clip_id] for clip in split.test], axis=0
    )
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[Any] = []
    for seed in config.seeds:
        noise_features = _real_noisy_tv_features(
            seed, config.noisy_tv_frames, featurizer, target_mean, target_std
        )
        seed_runs.append(
            _run_seed_real(seed, split, features_by_clip, noise_features, bed_config, train_density)
        )
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

    spread_block = _assemble_spread_diagnostic(seed_runs)

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
                "one real run of one frozen featurizer is a mechanics outcome; scientific confirmation "
                "needs the independent verifier plus at least three bias-independent reproductions and, "
                "for this three-featurizer family at n equals 5, cannot clear family-wise significance"
            ),
        },
    )

    truncations = [t.payload() for t in adapter.truncations()]
    dropped_onsets = sum(t["dropped_onsets_past_end"] for t in truncations)
    capped_clips = sum(1 for t in truncations if t["capped_by_max_frames"])

    body = artifact_envelope(
        schema=VARIANT_ARTIFACT_SCHEMA, report=report, seeds=config.seeds, per_seed=per_seed,
        stats=stats_block, controls=controls_block, flags=flags_block, verdict=verdict,
        featurizer={
            "family": VARIANT_ID,
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
            "d_feat": D_FEAT,
            "note": (
                "frozen zero-trained-parameter interchannel-coherence front-end; featurized inline (its "
                "digest differs from the frozen cache) and charged per arm from its honest per-frame count"
            ),
        }, gate={
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        }, receipt_payload=receipt,
        extra={
            "variant_id": VARIANT_ID,
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": beats_random,
            "variant": {
            "variant_id": VARIANT_ID,
            "kind": "featurizer_swap",
            "hypothesis": _variant_hypothesis(),
            "front_end": (
                "frozen magnitude-squared coherence between W and X, Y, Z per band plus DirAC directness "
                "per band; 64 bands by 4 spatial features equals 256, the exact width the sealed gate "
                "consumes, so the gate and its parameter count are unchanged"
            ),
            "only_featurizer_differs": True,
            "gate_unchanged": True,
            "featurizers_prereg_path": str(Path(featurizers_prereg_path)),
            "featurizers_prereg_canonical_sha256": prereg_digest,
            "fire_spread_diagnostic": spread_block,
        },
        "full_scale_anchors": {
            "c_train_flops": FULL_SCALE_C_TRAIN,
            "featurize_flops_24000_frames": FULL_SCALE_FEATURIZE,
            "featurize_flops_24000_frames_this_featurizer": FLOPS_PER_FRAME * 24_000,
            "downstream_flops_per_firing": bed_config.downstream_flops_per_firing,
            "break_even_frames_anchor": FULL_SCALE_C_TRAIN // bed_config.downstream_flops_per_firing,
        },
        "real_corpus": {
            "producer_schema": REAL_PRODUCER_SCHEMA,
            "variant_producer_schema": VARIANT_ARTIFACT_SCHEMA,
            "foa_root": str(Path(foa_root)),
            "metadata_root": str(Path(metadata_root)),
            "n_clips": len(adapter.clips()),
            "split_rooms": split.detail,
            "n_train_frames": seed_runs[0].train_frames,
            "n_test_clips": n_test_clips,
            "n_test_onsets": n_test_onsets,
            "n_test_frames": n_test_frames,
            "train_onset_density": round(float(train_density), 12),
            "operating_firing_fraction": round(float(operating_rate), 12),
            "truncation": {
                "clips_capped_by_max_frames": capped_clips,
                "onsets_dropped_past_audio_end": dropped_onsets,
                "max_frames": config.max_frames,
                "per_clip": truncations,
            },
        },
        "prereg": {
            "path": str(Path(featurizers_prereg_path)),
            "canonical_sha256": prereg_digest,
            "sesoi_f1": sesoi_f1,
            "provisional": False,
            "written_before_test_scores": True,
            "rebuilt_by_this_producer": False,
        },
        },
    )
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
            "featurizer_flops_per_frame": FLOPS_PER_FRAME,
            "candidate_max_lifecycle_flops": report.matched_budget.payload()["flops"],
            "spread": spread_block,
        },
    )
