"""Real-data producer for the frozen featurizer "superflux_spectral" on the STARSS23 ESCS bed.

This is a net-new, additive component. It runs the whole bed end to end on the REAL, MIT-licensed STARSS23
FOA subset with a NEW frozen zero-trained-parameter front-end (the SuperFlux onset detector) feeding the
UNCHANGED trained gate, and assembles a sealed ``proof/STARSS23_ESCS_BED_superflux_spectral.json`` with the
same shape as the committed real run. It changes NONE of the sealed scoring logic: the referee, the
matched-budget harness, the exact sign-flip statistics, and every control are imported unchanged, and the
per-seed training / firing / scoring pass ``real_artifact._run_seed_real`` is reused byte-for-byte. The
ONLY thing that differs from the committed real run is the frozen front-end: the gate reads the SuperFlux
features instead of the base half-wave-rectified log-mel flux.

The one accounting change the recipe mandates (gotcha F1): the FLOP model here charges the SuperFlux
front-end's OWN ``FLOPS_PER_FRAME`` (1,129,020, slightly above the base 1,121,340) to every arm equally,
NOT the base front-end's cost. The candidate and the rate-matched-random control still fire the same count
at byte-equal inference FLOPs, so the matched-budget invariant holds and any accuracy gap remains WHERE
compute is spent, not how much. Every arm total stays under the 6e10 lifecycle FLOP ceiling.

The SuperFlux front-end emits exactly 256 features per frame (64 mel-flux bins by 4 FOA channels), the
same dimensionality as the base front-end and the gate's hard-wired ``D_FEAT``, so the unchanged 264-input
gate consumes it directly with no projection, truncation, or padding and adds no trained parameter.

The SESOI and the whole analysis plan are read from the already-sealed featurizer preregistration
``proof/STARSS23_ESCS_BED_superflux_spectral.prereg.json``; this producer NEVER rebuilds or reweakens it.
The verdict is a mechanics outcome only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single run at n equals 5 across the
three-featurizer family can never promote.

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
from mop.science import (
    ArtifactResult,
    artifact_envelope,
    demonstration_receipt,
    finalize_artifact,
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
from .artifact import (
    FULL_SCALE_C_TRAIN,
    FULL_SCALE_TEST_FRAMES,
    PRIMARY_CONTROL,
    _SeedRun,
)
from .controls import at_chance
from .experiments import ONSET_BUDGET_POLICY
from .feature_cache import load_cached_corpus, load_or_build_cached_corpus
from .featurizer_superflux_spectral import FLOPS_PER_FRAME as SUPERFLUX_FLOPS_PER_FRAME
from .featurizer_superflux_spectral import SuperfluxSpectralFeaturizer
from .gate import FLOPS_PER_INFERENCE, OnlineState, training_flops
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    REAL_PRODUCER_SCHEMA,
    RealBedConfig,
    _real_noisy_tv_features,
    _run_seed_real,
)
from .referee import score_arm
from .schema import COLLAR_FRAMES
from .superflux_spectral_prereg import (
    DEFAULT_FEATURIZERS_PREREG_PATH,
    FEATURIZER_VARIANTS,
    FEATURIZERS_PREREG_SCHEMA,
)

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-superflux-spectral/v1"
VARIANT_ID = "superflux_spectral"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_superflux_spectral.json")

# Full-scale featurize anchor recomputed with the SuperFlux front-end's own per-frame cost.
SUPERFLUX_FULL_SCALE_FEATURIZE = SUPERFLUX_FLOPS_PER_FRAME * FULL_SCALE_TEST_FRAMES


class SuperfluxSpectralRefusal(ValueError):
    """Raised when the superflux_spectral producer cannot assemble a well-formed sealed artifact."""


def _variant_hypothesis() -> str:
    for entry in FEATURIZER_VARIANTS:
        if entry["variant_id"] == VARIANT_ID:
            return entry["hypothesis"]
    raise SuperfluxSpectralRefusal(f"featurizer {VARIANT_ID!r} is not in the sealed featurizer family")


def _read_sealed_featurizers_prereg(
    path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> dict[str, Any]:
    """Read the already-sealed featurizer preregistration. Never rebuilds or reweakens it."""

    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise SuperfluxSpectralRefusal(
            f"the sealed featurizer preregistration {prereg_path} is missing; seal it before the run"
        )
    body = json.loads(prereg_path.read_bytes().decode("utf-8"))
    if body.get("schema") != FEATURIZERS_PREREG_SCHEMA:
        raise SuperfluxSpectralRefusal(f"unexpected featurizer prereg schema {body.get('schema')!r}")
    ids = [entry["variant_id"] for entry in body.get("variants", [])]
    if VARIANT_ID not in ids:
        raise SuperfluxSpectralRefusal(f"{VARIANT_ID!r} is not preregistered in {prereg_path}")
    return body


# ---------------------------------------------------------------------------
# FLOP model: charges the SuperFlux front-end's OWN per-frame cost to every arm (recipe gotcha F1).
# ---------------------------------------------------------------------------


def _superflux_flop_model(kind: str, total_frames: int, train_frames: int, config: Any) -> FlopModel:
    """Full-lifecycle FLOP model charging the SuperFlux featurize cost, not the base front-end's.

    Every arm charges the SuperFlux featurize (candidate, rate_matched_random, always_on, best_single all
    pay ``SUPERFLUX_FLOPS_PER_FRAME * total_frames``), so matched budget still holds: candidate and
    rate_matched_random fire the same count at byte-equal inference FLOPs. Only the candidate charges the
    amortized training cost C_train; the controls learn nothing.
    """

    featurize = SUPERFLUX_FLOPS_PER_FRAME * total_frames
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
# Fire-spread diagnostics: adjacency fraction and distinct-onset true positives (same rule as the E1 wave).
# ---------------------------------------------------------------------------


def _adjacency_fraction(clip_fire_lists: list[list[int]], collar: int = COLLAR_FRAMES) -> float:
    """Pooled fraction of fires that have another fire within ``collar`` frames in the same clip.

    The committed null (base front-end) clustered roughly 42 percent of its fires adjacently on
    high-energy regions and recovered about 204 distinct onsets. A stronger front-end that spreads its
    novelty across separated onsets would lower this fraction and raise distinct-onset recovery.
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
    return {
        "fires": sum(len(fires) for fires in fire_lists),
        "adjacency_fraction": round(_adjacency_fraction(fire_lists, collar), 12),
        "distinct_onset_tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
    }


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _spread_from_per_seed(per_seed: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Summarize an arm's fire-spread across seeds at the operating point, from the per-seed clip blocks."""

    per_seed_rows: list[dict[str, Any]] = []
    for block in per_seed:
        pairs = [(clip["gt_onsets"], clip["fires"][arm]) for clip in block["clips"]]
        per_seed_rows.append(_arm_spread(pairs))
    return {
        "per_seed_fires": [row["fires"] for row in per_seed_rows],
        "per_seed_adjacency_fraction": [row["adjacency_fraction"] for row in per_seed_rows],
        "per_seed_distinct_onset_tp": [row["distinct_onset_tp"] for row in per_seed_rows],
        "mean_fires": round(_mean([row["fires"] for row in per_seed_rows]), 6),
        "mean_adjacency_fraction": round(_mean([row["adjacency_fraction"] for row in per_seed_rows]), 12),
        "mean_distinct_onset_tp": round(_mean([row["distinct_onset_tp"] for row in per_seed_rows]), 6),
    }


def _assemble_spread_diagnostic(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Fire-spread diagnostic at the operating point for the SuperFlux candidate and its matched control."""

    return {
        "definition": (
            "adjacency_fraction is the pooled fraction of test fires within the DCASE collar of another "
            "fire on the same clip; distinct_onset_tp is the pooled greedy one-to-one referee true "
            "positives at the operating budget. Both are computed from the SuperFlux run's own operating-"
            "point fires."
        ),
        "collar_frames": COLLAR_FRAMES,
        "candidate": _spread_from_per_seed(per_seed, ARM_CANDIDATE),
        "rate_matched_random": _spread_from_per_seed(per_seed, ARM_RATE_MATCHED_RANDOM),
        "committed_null_base_frontend_seed0_anchor": {
            "candidate_distinct_onset_tp": 204,
            "rate_matched_random_distinct_onset_tp": 237,
            "candidate_adjacency_fraction_approx": 0.42,
            "source": "docs/mixture_of_perspectives/26_escs_starss23_bed.md (base half-wave-rectified flux)",
        },
    }


# ---------------------------------------------------------------------------
# Assemble and seal the featurizer-variant artifact.
# ---------------------------------------------------------------------------


def build_superflux_spectral_artifact(
    *,
    timestamp: str,
    corpus: Any | None = None,
    cache_root: str | Path | None = None,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> ArtifactResult:
    """Run the superflux_spectral featurizer on the real corpus and assemble the sealed artifact.

    The SuperFlux features are read from the SuperFlux feature cache (built once if absent); the SESOI and
    analysis plan come from the already-sealed featurizer preregistration, never rebuilt here. The
    ``timestamp`` is passed by the caller and never read from the wall clock inside a sealed body.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    if corpus is None:
        if cache_root is None:
            corpus = load_or_build_cached_corpus(front_end="superflux")
        else:
            corpus = load_cached_corpus(front_end="superflux", cache_root=cache_root)
    split = corpus.split
    features_by_clip = corpus.features_by_clip

    prereg = _read_sealed_featurizers_prereg(featurizers_prereg_path)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])
    prereg_digest = str(prereg["canonical_sha256"])

    # Structural facts (label-only), for provenance and the operating-point rule.
    train_density = corpus.train_onset_density()
    n_test_clips = corpus.n_test_clips()
    n_test_onsets = corpus.n_test_onsets()
    n_test_frames = corpus.n_test_frames()
    if n_test_onsets == 0:
        raise SuperfluxSpectralRefusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # noisy-TV marginals: match the injected white-noise channel, featurized by the SAME SuperFlux front-
    # end, to the real SuperFlux test feature marginals, so only a novelty-chaser fires preferentially.
    featurizer = SuperfluxSpectralFeaturizer()
    pooled_test_features = np.concatenate(
        [features_by_clip[clip.clip_id] for clip in split.test], axis=0
    )
    target_mean = float(pooled_test_features.mean())
    target_std = float(pooled_test_features.std())

    started = time.perf_counter_ns()
    seed_runs: list[_SeedRun] = []
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
        flop_model=lambda kind: _superflux_flop_model(
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

    spread_block = _assemble_spread_diagnostic(per_seed)

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
            "variant_kind": "frozen_featurizer",
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

    body = artifact_envelope(
        schema=VARIANT_ARTIFACT_SCHEMA,
        report=report,
        seeds=config.seeds,
        per_seed=per_seed,
        stats=stats_block,
        controls=controls_block,
        flags=flags_block,
        verdict=verdict,
        featurizer={
            "variant_id": VARIANT_ID,
            "front_end": "superflux_spectral",
            "hypothesis": _variant_hypothesis(),
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": SUPERFLUX_FLOPS_PER_FRAME,
            "base_frontend_flops_per_frame": 1_121_340,
            "feature_dim": 256,
            "feature_cache_key": corpus.cache_key,
            "only_frozen_front_end_differs": True,
            "gate_unchanged": True,
            "no_projection_or_truncation": (
                "the SuperFlux front-end emits exactly 64 mel-flux bins by 4 channels = 256 features, the "
                "gate's hard-wired D_FEAT, so the unchanged gate consumes it with no adaptation and no "
                "trained parameter added"
            ),
            "note": (
                "featurized once with the SuperFlux front-end and cached; the FLOP ledger charges its own "
                "per-frame cost per arm from the cache count, so the cache is a wall-clock optimization "
                "and not a budget cut"
            ),
        },
        gate={
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
        receipt_payload=receipt,
        extra={
            "variant_id": VARIANT_ID,
            "variant_kind": "frozen_featurizer",
            "collar_frames": COLLAR_FRAMES,
            "primary_control": PRIMARY_CONTROL,
            "beats_rate_matched_random": beats_random,
            "fire_spread_diagnostic": spread_block,
            "full_scale_anchors": {
                "c_train_flops": FULL_SCALE_C_TRAIN,
                "featurize_flops_24000_frames": SUPERFLUX_FULL_SCALE_FEATURIZE,
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
            "spread": spread_block,
            "candidate_featurize_flops": SUPERFLUX_FLOPS_PER_FRAME * seed_runs[0].total_frames,
            "candidate_max_lifecycle_flops": max(
                point.candidate.max_lifecycle_flops() for point in budget_points
            ),
        },
    )
