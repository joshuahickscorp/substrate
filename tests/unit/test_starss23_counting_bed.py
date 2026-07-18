"""Unit tests for the STARSS23 ESCS concurrent-source-counting value-of-computation bed.

These exercise the net-new counting modules end to end without touching the sealed onset scoring path:
the frozen zero-parameter count featurizer and estimator, the trained count gate, the sealed
coasted-count-MAE referee (on a hand-built toy with a known answer), the matched-budget count harness, the
preregistered SESOI, and the real-data producer plus the independent verifier (seal, byte-reproducible
receipt, referee and stats reproduction, tamper detection, and the additive-only import boundary).

The end-to-end producer tests run on the REAL STARSS23 subset with a small frame cap and a reduced seed set
so they stay fast; when the real subset is absent they skip. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23 import count_harness as H
from mop.beds.starss23 import count_verifier as V
from mop.beds.starss23.count_estimator import (
    FLOPS_PER_REESTIMATE,
    FrozenCountEstimator,
)
from mop.beds.starss23.count_featurizer import (
    D_CFEAT,
    FLOPS_PER_FRAME_COUNT,
    FrozenCountFeaturizer,
)
from mop.beds.starss23.count_gate import (
    FLOPS_PER_INFERENCE,
    CountGate,
    CountOnlineState,
    voc_targets_from_count_track,
)
from mop.beds.starss23.count_labels import (
    COUNT_CEILING,
    CountClip,
    change_density,
    coast_from_zero_mae,
    count_track_from_metadata_text,
)
from mop.beds.starss23.count_prereg import PREREGISTERED_SESOI_MAE, build_count_prereg
from mop.beds.starss23.count_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealCountBedConfig,
    build_real_count_bed_artifact,
)
from mop.beds.starss23.count_referee import coast_emitted, mae_clip, score_arm
from mop.beds.starss23.gate import training_flops

FLOP_CEILING = 60_000_000_000
_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"


# ---------------------------------------------------------------------------
# Labels.
# ---------------------------------------------------------------------------


def test_count_track_distinct_tracks_and_tail_silence():
    # Two distinct (class, source) tracks active at frame 1; silence elsewhere including the tail.
    text = "0,1,1,10,0,100\n1,1,1,10,0,100\n1,2,3,20,0,100\n"
    track = count_track_from_metadata_text(text, n_frames=5)
    assert track.tolist() == [1, 2, 0, 0, 0]


def test_count_track_drops_rows_past_end_and_guards_ceiling():
    text = "0,1,1,10,0,100\n9,2,2,20,0,100\n"  # frame 9 is past n_frames
    track = count_track_from_metadata_text(text, n_frames=3)
    assert track.tolist() == [1, 0, 0]
    with pytest.raises(Exception):
        CountClip(
            clip_id="fold3_room0_mix000",
            room_id="room00",
            n_frames=1,
            audio_sha256="a" * 64,
            count_track=(COUNT_CEILING + 1,),
        )


def test_change_density_and_coast_from_zero():
    clip = CountClip(
        clip_id="fold3_room0_mix000",
        room_id="room00",
        n_frames=6,
        audio_sha256="a" * 64,
        count_track=(0, 1, 1, 2, 0, 0),
    )
    # changes at t=1 (0->1), t=3 (1->2), t=4 (2->0): 3 changes over 6 frames.
    assert clip.n_changes == 3
    assert clip.change_frames == (1, 3, 4)
    assert change_density([clip]) == pytest.approx(3 / 6)
    # mean|C_gt| = (0+1+1+2+0+0)/6
    assert coast_from_zero_mae([clip]) == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# Featurizer: zero trained parameters and byte-deterministic.
# ---------------------------------------------------------------------------


def test_featurizer_zero_param_and_deterministic():
    fz = FrozenCountFeaturizer()
    assert fz.n_params() == 0
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((4, 2400 * 12))
    f1 = fz.featurize(audio)
    f2 = fz.featurize(audio)
    assert f1.shape == (12, D_CFEAT)
    assert D_CFEAT == 256
    assert fz.feature_digest(f1) == fz.feature_digest(f2)
    assert fz.parameter_digest() == FrozenCountFeaturizer().parameter_digest()
    assert FLOPS_PER_FRAME_COUNT == 1_120_700


# ---------------------------------------------------------------------------
# Estimator: zero trained parameters, byte-reproducible, [0, 4] range, silence -> 0.
# ---------------------------------------------------------------------------


def test_estimator_zero_param_deterministic_range_and_silence():
    est = FrozenCountEstimator()
    assert est.n_params() == 0
    rng = np.random.default_rng(3)
    audio = rng.standard_normal((4, 2400 * 10))
    e1 = est.estimate_track(audio)
    e2 = est.estimate_track(audio)
    assert bool((e1 == e2).all())
    assert int(e1.min()) >= 0 and int(e1.max()) <= 4
    silence = np.zeros((4, 2400 * 4))
    assert est.estimate_track(silence).tolist() == [0, 0, 0, 0]
    assert est.flops_for_reestimations(5) == 5 * FLOPS_PER_REESTIMATE
    assert FLOPS_PER_REESTIMATE == 80_000


# ---------------------------------------------------------------------------
# Gate: 3193 params, few-KB state, no label online, paired-seed reproducibility, ponder lowers rate.
# ---------------------------------------------------------------------------


def test_gate_param_and_state_ceilings():
    gate = CountGate(seed=0)
    assert gate.n_params() == 3193
    assert gate.n_params() <= 4096
    assert CountOnlineState.state_bytes() <= 8192
    assert FLOPS_PER_INFERENCE == 6385


def test_gate_infer_takes_no_label_and_is_seed_reproducible():
    g1 = CountGate(seed=2)
    g2 = CountGate(seed=2)
    assert g1.parameter_digest() == g2.parameter_digest()
    features = np.random.default_rng(1).standard_normal(D_CFEAT)
    state = CountOnlineState.initial()
    # infer signature carries only features and self-state; no label argument exists.
    p1 = g1.infer(features, state)
    p2 = g2.infer(features, state)
    assert 0.0 <= p1 <= 1.0
    assert p1 == p2


def test_ponder_lowers_reestimation_rate():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 264))
    y = (rng.random(400) < 0.3).astype(np.float64)
    low = CountGate(seed=5)
    high = CountGate(seed=5)
    r_low = low.fit(x, y, epochs=6, learning_rate=0.1, ponder_lambda=0.0).final_reestimate_rate
    r_high = high.fit(x, y, epochs=6, learning_rate=0.1, ponder_lambda=0.5).final_reestimate_rate
    assert r_high <= r_low


def test_voc_targets_fire_near_changes_only():
    track = (0, 0, 1, 1, 1, 0)  # changes at t=2 and t=5
    targets = voc_targets_from_count_track(track, window=1)
    # window 1 around {2,5}: frames {1,2,3} and {4,5}
    assert targets.tolist() == [0, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Referee: coasted-count-MAE on a hand-built toy with a known answer.
# ---------------------------------------------------------------------------


def test_referee_toy_known_mae():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    reestimates = [1, 3]
    # emitted holds the last re-estimate from cold-start 0: 0,1,1,2,2
    assert coast_emitted(estimator, reestimates) == (0, 1, 1, 2, 2)
    abs_sum, n = mae_clip(gt, estimator, reestimates)
    assert (abs_sum, n) == (2, 5)  # only frame 4 is wrong (emitted 2 vs gt 0)


def test_referee_always_on_equals_mean_abs_e_minus_gt_and_never_update_equals_mean_gt():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    # always-on re-estimates every frame: emitted == estimator, MAE == mean|E - gt|
    ao_abs, ao_n = mae_clip(gt, estimator, list(range(len(gt))))
    assert ao_abs == sum(abs(estimator[t] - gt[t]) for t in range(len(gt)))
    # never-update re-estimates nothing: emitted == 0 everywhere, MAE == mean|gt|
    nu_abs, nu_n = mae_clip(gt, estimator, [])
    assert nu_abs == sum(gt)


def test_referee_pooling_micro_averages_across_clips():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    reestimates = [1, 3]
    score = score_arm([(gt, estimator, reestimates), (gt, estimator, reestimates)])
    assert score.abs_error_sum == 4 and score.n_frames == 10
    assert score.mae == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Harness: matched budget holds, ceiling holds, dominance minimizes MAE.
# ---------------------------------------------------------------------------


def _flop_model(kind, total_frames, train_frames):
    runs_gate = kind in (H.ARM_CANDIDATE, H.ARM_RATE_MATCHED_RANDOM)
    return H.FlopModel(
        featurize_flops=FLOPS_PER_FRAME_COUNT * total_frames,
        gate_infer_flops=FLOPS_PER_INFERENCE * total_frames if runs_gate else 0,
        downstream_flops_per_firing=FLOPS_PER_REESTIMATE,
        train_flops=training_flops(train_frames, 8) if kind == H.ARM_CANDIDATE else 0,
    )


def _arm(kind, mae_by_seed, k_by_seed, total_frames, train_frames, seeds):
    return H.CountArm(
        name=f"{kind}",
        kind=kind,
        total_frames=total_frames,
        params=3193 if kind == H.ARM_CANDIDATE else 0,
        flop_model=_flop_model(kind, total_frames, train_frames),
        seed_results=tuple(
            H.CountArmSeedResult(seed=s, mae=mae_by_seed[i], reestimations=k_by_seed[i])
            for i, s in enumerate(seeds)
        ),
    )


def test_harness_matched_budget_and_ceiling_and_dominance():
    seeds = (0, 1, 2, 3, 4)
    total_frames = 22569
    train_frames = 25000
    k = [1128] * 5
    cand = _arm(H.ARM_CANDIDATE, [0.30] * 5, k, total_frames, train_frames, seeds)
    rmr = _arm(H.ARM_RATE_MATCHED_RANDOM, [0.34] * 5, k, total_frames, train_frames, seeds)
    ao = _arm(H.ARM_ALWAYS_ON, [0.25] * 5, [total_frames] * 5, total_frames, train_frames, seeds)
    nu = _arm(H.ARM_NEVER_UPDATE, [1.25] * 5, [0] * 5, total_frames, train_frames, seeds)
    # Matched ex-training must pass with equal K and byte-equal inference FLOPs.
    H.assert_matched_ex_training(cand, rmr)
    point = H.CountBudgetPoint("rate_0.05", cand, rmr, ao, nu)
    point.certify()
    for arm in point.arms():
        assert arm.max_lifecycle_flops() <= FLOP_CEILING
    report = H.run_matched_budget([point], wall_ns=1, source_kind="real")
    assert report.candidate_strictly_dominates_rate_matched_random is True
    assert report.verdict == "mechanics-ok"
    assert report.activation_allowed is False
    assert report.scientific_promotion is False
    assert report.independent_scientific_confirmation is False


def test_harness_matched_budget_refuses_uncharged_training_and_k_mismatch():
    seeds = (0, 1)
    total_frames = 1000
    cand = _arm(H.ARM_CANDIDATE, [0.3, 0.3], [50, 50], total_frames, 1000, seeds)
    # A candidate whose flop model charges no training is refused.
    cand_no_train = H.CountArm(
        name="candidate",
        kind=H.ARM_CANDIDATE,
        total_frames=total_frames,
        params=3193,
        flop_model=H.FlopModel(featurize_flops=1, gate_infer_flops=1, downstream_flops_per_firing=1, train_flops=0),
        seed_results=cand.seed_results,
    )
    rmr = _arm(H.ARM_RATE_MATCHED_RANDOM, [0.3, 0.3], [50, 50], total_frames, 1000, seeds)
    with pytest.raises(H.CountUnchargedTraining):
        H.assert_matched_ex_training(cand_no_train, rmr)
    # A control whose K differs from the candidate is refused.
    rmr_bad = _arm(H.ARM_RATE_MATCHED_RANDOM, [0.3, 0.3], [40, 50], total_frames, 1000, seeds)
    with pytest.raises(H.CountBudgetMismatch):
        H.assert_matched_ex_training(cand, rmr_bad)


def test_harness_refuses_ceiling_exceeded():
    seeds = (0, 1)
    huge = H.CountArm(
        name="candidate",
        kind=H.ARM_CANDIDATE,
        total_frames=1000,
        params=3193,
        flop_model=H.FlopModel(
            featurize_flops=FLOP_CEILING, gate_infer_flops=1, downstream_flops_per_firing=1, train_flops=1
        ),
        seed_results=tuple(H.CountArmSeedResult(seed=s, mae=0.1, reestimations=1) for s in seeds),
    )
    with pytest.raises(H.CountCeilingExceeded):
        H.assert_within_ceiling(huge)


def test_pareto_minimizes_both_flops_and_mae():
    pts = [
        H.CountComputePoint("a", "x", flops=10.0, mae=0.5, params=0, reestimations=1.0),
        H.CountComputePoint("b", "y", flops=20.0, mae=0.6, params=0, reestimations=1.0),  # dominated
        H.CountComputePoint("c", "z", flops=30.0, mae=0.3, params=0, reestimations=1.0),
    ]
    frontier = {p.budget_id for p in H.pareto_frontier(pts)}
    assert frontier == {"a", "c"}


# ---------------------------------------------------------------------------
# Preregistration: SESOI is 0.02, sealed, with quantified rationale numbers.
# ---------------------------------------------------------------------------


def test_prereg_sesoi_and_rationale_numbers():
    body = build_count_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        n_test_clips=21,
        n_test_changes=916,
        n_test_frames=22569,
        train_change_density=0.052,
        coast_from_zero_mae=1.2552,
    )
    assert body["sesoi"]["sesoi_mae"] == PREREGISTERED_SESOI_MAE == 0.02
    assert body["preregistered_before_reading_test_scores"] is True
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert "canonical_sha256" in body
    cb = body["sesoi"]["cost_benefit"]
    # 0.02 is ~451x the per-frame granularity floor (1 / 22569).
    assert round(0.02 / cb["per_frame_granularity"]) == 451
    # one test clip's change mass is about 0.0238 pooled MAE, so 0.02 sits at about one clip.
    assert cb["one_clip_change_mass_mae"] == pytest.approx(0.0238, abs=5e-4)
    assert body["sign_flip_test_plan"]["min_one_sided_p"] == pytest.approx(1 / 32)
    assert body["sign_flip_test_plan"]["two_sided_alpha_reachable"] is False


# ---------------------------------------------------------------------------
# Verifier import boundary: no producer, referee, stats, harness, or mop imports.
# ---------------------------------------------------------------------------


def test_verifier_imports_no_producer_or_mop_code():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("mop", "count_referee", "count_producer", "count_harness", "stats", "harness", "referee")
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), name
    # It uses only the standard library.
    assert set(n.split(".")[0] for n in imported) <= {"json", "hashlib", "itertools", "dataclasses", "__future__"}


# ---------------------------------------------------------------------------
# End-to-end producer + verifier on the REAL subset (small, fast config).
# ---------------------------------------------------------------------------


_SMALL_CONFIG = RealCountBedConfig(seeds=(0, 1, 2), target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=100)


@pytest.fixture(scope="module")
def real_count_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("count_bed")
    built = build_real_count_bed_artifact(
        timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json"
    )
    return built


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_artifact_within_ceiling(real_count_artifact):
    art = real_count_artifact.artifact
    assert "seal" in art and isinstance(art["seal"], str) and len(art["seal"]) == 64
    assert art["schema"] == "mop-starss23-escs-count-bed/v1"
    assert art["source_kind"] == "real" and art["rights_clean"] is True
    assert art["verdict"] in ("mechanics-ok", "null")
    assert art["flags"] == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    # Every arm total stays under the 6e10 ceiling.
    for arm in art["harness"]["arm_summaries"]:
        assert arm["max_lifecycle_flops"] <= FLOP_CEILING
    assert art["matched_budget"]["flops"] <= FLOP_CEILING


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_receipt_and_seal_byte_reproducible(tmp_path):
    # The prereg path is recorded in the sealed body (as in the onset bed), so both runs must write the
    # same path for the seal to be byte-reproducible; the computation itself is fully deterministic.
    prereg_path = tmp_path / "prereg.json"
    a = build_real_count_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    b = build_real_count_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    assert a.artifact["seal"] == b.artifact["seal"]
    assert a.artifact["demonstration_receipt"] == b.artifact["demonstration_receipt"]
    assert a.prereg["canonical_sha256"] == b.prereg["canonical_sha256"]


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_reproduces_referee_and_stats(real_count_artifact):
    result = V.verify_count_artifact(real_count_artifact.artifact)
    assert result.seal_intact is True
    assert result.scores_reproduced is True
    assert result.stats_reproduced is True
    assert result.honesty_ok is True
    assert result.independent_referee_reproduction is True
    # One real run can reproduce the referee but never self-confirms (needs >= 3 reproductions).
    assert result.independent_scientific_confirmation is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_score(real_count_artifact):
    art = copy.deepcopy(real_count_artifact.artifact)
    # Corrupt a stored candidate MAE and re-seal so the seal passes but the re-derivation catches it.
    art["per_seed"][0]["arm_scores"]["candidate"]["mae"] += 0.5
    body = {k: v for k, v in art.items() if k != "seal"}
    art["seal"] = V._canonical_sha256(body)
    result = V.verify_count_artifact(art)
    assert result.seal_intact is True
    assert result.scores_reproduced is False
    assert result.independent_referee_reproduction is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_reestimates_and_estimator_track(real_count_artifact):
    # Tampering with the candidate re-estimation frames breaks the budget match and the MAE.
    art = copy.deepcopy(real_count_artifact.artifact)
    clip0 = art["per_seed"][0]["clips"][0]
    if clip0["reestimate_frames"]["candidate"]:
        clip0["reestimate_frames"]["candidate"] = clip0["reestimate_frames"]["candidate"][:-1]
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    assert V.verify_count_artifact(art).scores_reproduced is False

    # Tampering with the shared estimator track changes always-on and candidate MAE.
    art2 = copy.deepcopy(real_count_artifact.artifact)
    some_clip = next(iter(art2["corpus_tracks"]))
    track = art2["corpus_tracks"][some_clip]["estimator_track"]
    track[0] = track[0] + 3
    art2["seal"] = V._canonical_sha256({k: v for k, v in art2.items() if k != "seal"})
    assert V.verify_count_artifact(art2).scores_reproduced is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_broken_seal(real_count_artifact):
    art = copy.deepcopy(real_count_artifact.artifact)
    art["seal"] = "0" * 64
    result = V.verify_count_artifact(art)
    assert result.seal_intact is False
    assert result.independent_referee_reproduction is False
