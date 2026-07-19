"""Real-data producer for the FROZEN featurizer swap "spatial_doa" on the STARSS23 ESCS bed.

This is a net-new, additive component. It runs the spatial-DOA frozen featurizer end to end on the REAL,
MIT-licensed STARSS23 FOA subset and assembles a sealed ``proof/STARSS23_ESCS_BED_spatial_doa.json`` with
the same shape as the committed ``proof/STARSS23_ESCS_BED.json``. It changes NONE of the sealed scoring
logic: the referee, the matched-budget harness and its FLOP accounting, the exact sign-flip statistics,
and every control are imported unchanged; the value-of-computation training-target assembly, the causal
p_fire pass, the per-seed run, and the room-disjoint fold split are imported unchanged from the committed
producers. The ONE thing that differs from the committed real run is the frozen FRONT-END: the features
the gate reads are per-band active-intensity direction-of-arrival features, not log-mel spectral flux.

Because the featurizer's per-frame FLOP cost differs from the frozen front-end, this producer builds its
OWN FLOP model charging ``featurizer_spatial_doa.FLOPS_PER_FRAME`` for the featurize term of EVERY arm
(candidate, rate-matched-random, always-on, best-single), so matched budget still holds (candidate and
rate-matched-random remain byte-equal in inference FLOPs) and every arm stays under the 6e10 ceiling. The
gate, its 3193-parameter count, its 4096 ceiling, and its amortized C_train are the committed anchors,
unchanged, because the featurizer emits exactly 256 features and the gate is not touched.

The SESOI and the whole analysis plan are read from the already-sealed featurizer preregistration
``proof/STARSS23_ESCS_BED_spatial_doa.prereg.json``; this producer NEVER rebuilds or reweakens it. The
verdict is a mechanics outcome only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, and a single run at n equals 5 across the
three-featurizer family can never promote.

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
from mop.science.budget import (
    ARM_ALWAYS_ON,
    ARM_BEST_SINGLE,
    ARM_CANDIDATE,
    ARM_RATE_MATCHED_RANDOM,
    Arm,
    BudgetPoint,
    FlopModel,
    SeedResult,
    run_matched_budget,
)
from mop.science.statistics import exact_sign_flip
from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING, STAGE3_FORCING_NULL
from .artifact import (
    DOWNSTREAM_FLOPS_PER_FIRING,
    FULL_SCALE_C_TRAIN,
    PRIMARY_CONTROL,
    STAGE,
    STAGE3_REQUIREMENT_ID,
    _SeedRun,
)
from .experiments import ONSET_BUDGET_POLICY
from .feature_cache import CachedCorpus, load_or_build_cached_corpus
from .featurizer_spatial_doa import D_FEAT, FLOPS_PER_FRAME, SpatialDoaFeaturizer
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
from .spatial_doa_prereg import DEFAULT_FEATURIZERS_PREREG_PATH, FEATURIZERS, FEATURIZERS_PREREG_SCHEMA

VARIANT_ARTIFACT_SCHEMA = "mop-starss23-escs-bed-spatial-doa/v1"
FEATURIZER_ID = "spatial_doa"

DEFAULT_VARIANT_ARTIFACT_PATH = Path("proof/STARSS23_ESCS_BED_spatial_doa.json")

# Full-scale featurize anchor recomputed for the spatial-DOA front-end (recorded for provenance).
FULL_SCALE_TEST_FRAMES = 24_000
FULL_SCALE_FEATURIZE = FLOPS_PER_FRAME * FULL_SCALE_TEST_FRAMES


class SpatialDoaRefusal(ValueError):
    """Raised when the spatial_doa producer cannot assemble a well-formed sealed artifact."""


def _featurizer_hypothesis() -> str:
    for entry in FEATURIZERS:
        if entry["featurizer_id"] == FEATURIZER_ID:
            return entry["hypothesis"]
    raise SpatialDoaRefusal(f"featurizer {FEATURIZER_ID!r} is not in the sealed featurizer family")


def _read_sealed_featurizers_prereg(
    path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> dict[str, Any]:
    """Read the already-sealed featurizer preregistration. Never rebuilds or reweakens it."""

    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise SpatialDoaRefusal(
            f"the sealed featurizer preregistration {prereg_path} is missing; seal it before the run"
        )
    body = json.loads(prereg_path.read_bytes().decode("utf-8"))
    if body.get("schema") != FEATURIZERS_PREREG_SCHEMA:
        raise SpatialDoaRefusal(f"unexpected featurizer prereg schema {body.get('schema')!r}")
    ids = [entry["featurizer_id"] for entry in body.get("featurizers", [])]
    if FEATURIZER_ID not in ids:
        raise SpatialDoaRefusal(f"{FEATURIZER_ID!r} is not preregistered in {prereg_path}")
    return body


# ---------------------------------------------------------------------------
# The spatial-DOA FLOP model: featurize is charged at the DOA per-frame cost for EVERY arm (gotcha F1).
# ---------------------------------------------------------------------------


def _flop_model(kind: str, total_frames: int, train_frames: int, epochs: int) -> FlopModel:
    """Full-lifecycle FLOP model for one arm, charging the spatial-DOA featurize cost on every arm.

    Every arm charges the same featurize FLOPs (the front-end runs regardless of the firing policy), so
    matched budget still holds: candidate and rate-matched-random have byte-equal inference FLOPs. Only
    the candidate charges the amortized training cost C_train; only candidate and rate-matched-random run
    the gate at inference.
    """

    featurize = FLOPS_PER_FRAME * total_frames
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    gate_infer = FLOPS_PER_INFERENCE * total_frames if runs_gate else 0
    train = training_flops(train_frames, epochs) if kind == ARM_CANDIDATE else 0
    return FlopModel(
        featurize_flops=featurize,
        gate_infer_flops=gate_infer,
        downstream_flops_per_firing=DOWNSTREAM_FLOPS_PER_FIRING,
        train_flops=train,
    )


def _build_budget_points(seed_runs: list[_SeedRun], epochs: int) -> list[BudgetPoint]:
    """Assemble the harness budget points with the spatial-DOA FLOP model on every arm."""

    total_frames = seed_runs[0].total_frames
    train_frames = seed_runs[0].train_frames
    params = {ARM_CANDIDATE: seed_runs[0].gate_params}
    budget_points: list[BudgetPoint] = []
    for budget_id in seed_runs[0].per_budget:
        arms: dict[str, Arm] = {}
        for kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE):
            seed_results = tuple(
                SeedResult(
                    seed=run.seed,
                    metric_value=run.per_budget[budget_id]["arm_scores"][kind]["f1"],
                    actions=run.per_budget[budget_id]["firings"][kind],
                )
                for run in seed_runs
            )
            arms[kind] = Arm(
                policy=ONSET_BUDGET_POLICY,
                name=f"{kind}@{budget_id}",
                kind=kind,
                total_frames=total_frames,
                params=params.get(kind, 0),
                flop_model=_flop_model(kind, total_frames, train_frames, epochs),
                seed_results=seed_results,
            )
        budget_points.append(
            BudgetPoint(
                policy=ONSET_BUDGET_POLICY,
                budget_id=budget_id,
                candidate=arms[ARM_CANDIDATE],
                rate_matched_random=arms[ARM_RATE_MATCHED_RANDOM],
                always_on=arms[ARM_ALWAYS_ON],
                reference=arms[ARM_BEST_SINGLE],
            )
        )
    return budget_points


# ---------------------------------------------------------------------------
# Fire-spread diagnostics: adjacency and distinct-onset true positives at the operating point.
# ---------------------------------------------------------------------------


def _adjacency_fraction(clip_fire_lists: list[list[int]], collar: int = COLLAR_FRAMES) -> float:
    """Pooled fraction of fires that have another fire within ``collar`` frames in the same clip."""

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


def _arm_spread_from_seed_block(per_seed_block: dict[str, Any], arm: str) -> dict[str, Any]:
    """Pooled fire count, adjacency fraction, and distinct-onset true positives for one arm at operating."""

    clips = per_seed_block["clips"]
    pairs = [(list(clip["gt_onsets"]), list(clip["fires"][arm])) for clip in clips]
    fire_lists = [fires for _gt, fires in pairs]
    score = score_arm(pairs, COLLAR_FRAMES)
    return {
        "fires": sum(len(fires) for fires in fire_lists),
        "adjacency_fraction": round(_adjacency_fraction(fire_lists), 12),
        "distinct_onset_tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
    }


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _assemble_spread_diagnostic(seed_runs: list[_SeedRun]) -> dict[str, Any]:
    """Summarize the per-seed operating-point fire-spread for the candidate and rate-matched-random."""

    def _summary(arm: str) -> dict[str, Any]:
        per_seed = [_arm_spread_from_seed_block(run.per_seed_block, arm) for run in seed_runs]
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
            "positives at the operating budget, under the spatial-DOA front-end"
        ),
        "collar_frames": COLLAR_FRAMES,
        "candidate": _summary(ARM_CANDIDATE),
        "rate_matched_random": _summary(ARM_RATE_MATCHED_RANDOM),
        "committed_null_seed0_anchor": {
            "candidate_distinct_onset_tp": 204,
            "rate_matched_random_distinct_onset_tp": 237,
            "candidate_adjacency_fraction_approx": 0.42,
            "source": "docs/mixture_of_perspectives/26_escs_starss23_bed.md (log-mel flux front-end)",
        },
    }


# ---------------------------------------------------------------------------
# Assemble and seal the spatial_doa artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialDoaArtifact:
    """The assembled sealed spatial-DOA featurizer artifact plus the mechanics-only detail."""

    artifact: dict[str, Any]
    verdict: str
    detail: dict[str, Any]

    @property
    def seal(self) -> str:
        return self.artifact["seal"]


def build_spatial_doa_artifact(
    *,
    timestamp: str,
    corpus: CachedCorpus | None = None,
    cache_root: str | Path | None = None,
    config: RealBedConfig | None = None,
    featurizers_prereg_path: str | Path = DEFAULT_FEATURIZERS_PREREG_PATH,
) -> SpatialDoaArtifact:
    """Run the spatial-DOA featurizer on the real corpus and assemble the sealed artifact.

    The SESOI and the analysis plan come from the already-sealed featurizer preregistration; this producer
    never rebuilds or reweakens it. ``timestamp`` is passed by the caller and never read from the wall
    clock inside a sealed body. The frozen spatial-DOA featurizer is featurized once and cached; the FLOP
    ledger still charges it per arm from the honest per-frame count.
    """

    config = config or RealBedConfig()
    bed_config = config.bed_config()
    if corpus is None:
        kwargs: dict[str, Any] = {} if cache_root is None else {"cache_root": cache_root}
        corpus = load_or_build_cached_corpus(front_end="spatial_doa", **kwargs)
    split = corpus.split
    features_by_clip = corpus.features_by_clip

    prereg = _read_sealed_featurizers_prereg(featurizers_prereg_path)
    sesoi_f1 = float(prereg["sesoi"]["sesoi_f1"])
    prereg_digest = str(prereg["canonical_sha256"])

    # Structural facts (label-only), used for provenance and the operating-point rule.
    train_density = corpus.train_onset_density()
    n_test_clips = corpus.n_test_clips()
    n_test_onsets = corpus.n_test_onsets()
    n_test_frames = corpus.n_test_frames()
    if n_test_onsets == 0:
        raise SpatialDoaRefusal("the real test split carries no onsets to score")
    operating_rate = min(bed_config.target_rates, key=lambda r: abs(r - train_density))

    # noisy-TV marginals: match the injected white-noise channel to the real test feature marginals under
    # the spatial-DOA front-end.
    featurizer = SpatialDoaFeaturizer()
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

    budget_points = _build_budget_points(seed_runs, bed_config.epochs)
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
    stats_block = {
        "deltas": [float(value) for value in deltas],
        "t_obs": float(sign_flip.mean_delta),
        "one_sided_p": float(sign_flip.one_sided_p),
        "n_permutations": int(sign_flip.permutations),
        "two_sided_005_reachable": bool(sign_flip.two_sided_alpha_reachable),
        "sesoi_f1": sesoi_f1,
        "sesoi_provisional": False,
        "mean_delta_exceeds_sesoi": mean_delta_exceeds_sesoi,
        "beats_rate_matched_random": beats_random,
        "claim_verb": "consistent with",
        "experimental_unit": "clip",
        "frame_or_clip_bootstrap_allowed": False,
        "prereg_canonical_sha256": prereg_digest,
    }

    n_runs = len(seed_runs)
    mean_noise_rate = math.fsum(run.noisy_tv["firing_rate_on_noise"] for run in seed_runs) / n_runs
    mean_base_rate = math.fsum(run.noisy_tv["base_rate"] for run in seed_runs) / n_runs
    from .controls import at_chance

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
            "featurizer_id": FEATURIZER_ID,
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

    body: dict[str, Any] = {
        "schema": VARIANT_ARTIFACT_SCHEMA,
        "stage": STAGE,
        "bed_id": BED_ID,
        "featurizer_id": FEATURIZER_ID,
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
            "front_end": "spatial_doa_active_intensity",
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "d_feat": D_FEAT,
            "flops_per_frame": FLOPS_PER_FRAME,
            "feature_cache_key": corpus.cache_key,
            "note": (
                "frozen zero-trained-parameter active-intensity DOA front-end (64 bands x [dir_x, dir_y, "
                "dir_z, diffuseness] = 256); featurized once and cached; the FLOP ledger charges it per "
                "arm from the honest per-frame count, so caching is not a budget cut"
            ),
        },
        "gate": {
            "params": seed_runs[0].gate_params,
            "param_ceiling": 4096,
            "state_bytes": OnlineState.state_bytes(),
            "flops_per_inference": FLOPS_PER_INFERENCE,
        },
        "featurizer_swap": {
            "featurizer_id": FEATURIZER_ID,
            "hypothesis": _featurizer_hypothesis(),
            "front_end": (
                "per-band active-intensity direction of arrival (az/el direction cosines) + "
                "DirAC diffuseness"
            ),
            "replaces": "half-wave-rectified log-mel spectral flux",
            "only_front_end_differs": True,
            "gate_unchanged": True,
            "output_dim": D_FEAT,
            "output_dim_note": (
                "emits exactly 256 features natively, so the unchanged gate consumes it with no "
                "projection and no truncation; the gate's hardcoded length-256 feature contract holds"
            ),
            "featurizers_prereg_path": str(Path(featurizers_prereg_path)),
            "featurizers_prereg_canonical_sha256": prereg_digest,
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
            "featurizer_producer_schema": VARIANT_ARTIFACT_SCHEMA,
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
        "demonstration_receipt": receipt.payload(),
    }
    body["seal"] = canonical_sha256(body)

    return SpatialDoaArtifact(
        artifact=body,
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
            "candidate_max_lifecycle_flops": int(report.matched_budget.flops),
            "flop_ceiling": FLOP_CEILING,
            "spread": spread_block,
        },
    )


def write_variant_artifact(
    artifact: dict[str, Any], out_path: str | Path = DEFAULT_VARIANT_ARTIFACT_PATH
) -> Path:
    """Write the sealed spatial-DOA artifact as canonical JSON bytes so its on-disk digest is reproducible."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact))
    return path
