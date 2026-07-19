
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23 import doa_verifier as V
from mop.beds.starss23.doa_estimator import FLOPS_PER_REESTIMATE, FrozenDoaEstimator
from mop.beds.starss23.doa_featurizer import D_FEAT_DOA, DoaFeaturizer
from mop.beds.starss23.doa_featurizer import FLOPS_PER_FRAME as FEATURIZER_FLOPS_PER_FRAME
from mop.beds.starss23.doa_gate import (
    ARCH_A_ID,
    ARCH_B_ID,
    FLOPS_PER_INFERENCE_ARCH_A,
    FLOPS_PER_INFERENCE_ARCH_B,
    DoaGateArchA,
    DoaGateArchB,
    DoaOnlineState,
    build_gate,
    param_count_arch_a,
    param_count_arch_b,
    training_flops_arch_a,
    training_flops_arch_b,
)
from mop.beds.starss23.doa_labels import (
    DoaClip,
    doa_voc_targets_from_track,
    great_circle_degrees,
    great_circle_degrees_batch,
)
from mop.beds.starss23.doa_prereg import (
    DoaClipLabelFact,
    DoaPreregRefusal,
    build_doa_prereg,
    compute_doa_sesoi_deg,
)
from mop.beds.starss23.doa_producer import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    RealDoaBedConfig,
    build_real_doa_bed_artifact,
)
from mop.beds.starss23.doa_referee import (
    DOA_COLD_START,
    DoaRefereeRefusal,
    coast_emitted_direction,
    exact_sign_flip_over_clips,
    macro_score_arm,
    mae_deg_clip,
    pooled_score_arm,
    room_majority_collapse,
)
from mop.beds.starss23.experiments import DOA_BUDGET_POLICY
from mop.science import budget as H

FLOP_CEILING = 60_000_000_000
_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"




def test_great_circle_degrees_symmetric_identity_and_antipodal():
    assert great_circle_degrees(30.0, 10.0, 30.0, 10.0) == pytest.approx(0.0, abs=1e-6)
    assert great_circle_degrees(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0, abs=1e-6)
    assert great_circle_degrees(10.0, -20.0, 50.0, 35.0) == pytest.approx(
        great_circle_degrees(50.0, 35.0, 10.0, -20.0), abs=1e-9
    )


def test_great_circle_degrees_batch_matches_scalar():
    a = np.array([[0.0, 0.0], [10.0, 20.0], [-170.0, 5.0]])
    b = np.array([[0.0, 0.0], [15.0, -5.0], [175.0, -5.0]])
    batch = great_circle_degrees_batch(a, b)
    for i in range(a.shape[0]):
        assert batch[i] == pytest.approx(great_circle_degrees(*a[i], *b[i]), abs=1e-9)


def test_doa_clip_change_frames_and_voc_targets():
    track = (
        None,
        (0.0, 0.0),  # t=1: first active -> change
        (2.0, 0.0),  # t=2: jump 2deg < threshold -> not a change
        (40.0, 0.0),  # t=3: big jump -> change
        None,  # t=4: silence
        (41.0, 0.0),  # t=5: resumes after silence gap -> change (jump size irrelevant to the rule)
    )
    clip = DoaClip(clip_id="fold4_room10_mix001", room_id="room10", n_frames=6, audio_sha256="a" * 64, doa_track=track)
    assert clip.active_frames == (1, 2, 3, 5)
    assert clip.n_active_frames == 4
    assert clip.change_frames == (1, 3, 5)
    assert clip.n_changes == 3

    targets = doa_voc_targets_from_track(track, window=1)
    assert targets.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]




def test_featurizer_zero_param_and_deterministic():
    fz = DoaFeaturizer()
    assert fz.n_params() == 0
    rng = np.random.default_rng(11)
    audio = rng.standard_normal((4, 2400 * 9))
    f1 = fz.featurize(audio)
    f2 = fz.featurize(audio)
    assert f1.shape == (9, D_FEAT_DOA) == (9, 256)
    assert np.array_equal(f1, f2)
    assert (f1 >= 0.0).all()  # full-wave rectified absolute flux is nonnegative by construction
    assert fz.feature_digest(f1) == fz.feature_digest(f2)
    assert fz.parameter_digest() == DoaFeaturizer().parameter_digest()
    assert FEATURIZER_FLOPS_PER_FRAME == 1_127_761
    assert fz.flops_for_frames(9) == 9 * FEATURIZER_FLOPS_PER_FRAME


def test_featurizer_silence_has_zero_flux_everywhere():
    fz = DoaFeaturizer()
    silence = np.zeros((4, 2400 * 4))
    features = fz.featurize(silence)
    assert features.shape == (4, 256)
    assert np.allclose(features, 0.0, atol=1e-8)




def test_estimator_zero_param_deterministic_and_silence_is_cold_start():
    est = FrozenDoaEstimator()
    assert est.n_params() == 0
    rng = np.random.default_rng(4)
    audio = rng.standard_normal((4, 2400 * 5))
    t1 = est.estimate_track(audio)
    t2 = est.estimate_track(audio)
    assert t1.shape == (5, 2)
    assert np.array_equal(t1, t2)
    silence = np.zeros((4, 2400 * 3))
    silent_track = est.estimate_track(silence)
    assert np.allclose(silent_track, np.array(DOA_COLD_START))
    assert FLOPS_PER_REESTIMATE == 33_687
    assert est.flops_for_reestimations(7) == 7 * FLOPS_PER_REESTIMATE




def test_both_architectures_under_param_ceiling_with_exact_counts():
    assert param_count_arch_a() == 3193 <= 4096
    assert param_count_arch_b() == 1639 <= 4096
    assert FLOPS_PER_INFERENCE_ARCH_A == 6385
    assert FLOPS_PER_INFERENCE_ARCH_B == 3277
    ga = DoaGateArchA(seed=0)
    gb = DoaGateArchB(seed=0)
    assert ga.n_params() == 3193
    assert gb.n_params() == 1639
    assert DoaOnlineState.state_bytes() == 64 <= 8192


def test_gate_seed_reproducible_and_no_label_signature():
    g1 = build_gate(ARCH_A_ID, seed=2)
    g2 = build_gate(ARCH_A_ID, seed=2)
    assert g1.parameter_digest() == g2.parameter_digest()
    features = np.random.default_rng(1).standard_normal(D_FEAT_DOA)
    features = np.abs(features)  # DoaFeaturizer output is nonnegative
    state = DoaOnlineState.initial()
    p1 = g1.infer(features, state)
    p2 = g2.infer(features, state)
    assert 0.0 <= p1 <= 1.0
    assert p1 == p2
    for architecture in (ARCH_A_ID, ARCH_B_ID):
        gate = build_gate(architecture, seed=0)
        decided, p = gate.decide(features, state)
        assert isinstance(decided, bool) and 0.0 <= p <= 1.0


def test_training_flops_anchors_match_design():
    assert training_flops_arch_a(54_000) == 8_274_960_000
    assert training_flops_arch_b(54_000) == 4_246_992_000




def test_referee_toy_known_mae_and_coasting():
    gt = np.array([[10.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 0.0]])
    estimator = np.array([[10.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 0.0]])
    active_mask = np.array([True, True, True, True])
    emitted = coast_emitted_direction(estimator, [0, 2])
    assert np.allclose(emitted, gt, atol=1e-6)
    mae, n_active = mae_deg_clip(gt, estimator, [0, 2], active_mask)
    assert n_active == 4
    assert mae == pytest.approx(0.0, abs=1e-4)

    mae_stale, _ = mae_deg_clip(gt, estimator, [0], active_mask)
    assert mae_stale == pytest.approx((0.0 + 0.0 + 10.0 + 10.0) / 4.0, abs=1e-3)


def test_referee_never_update_uses_cold_start_and_active_frames_only_are_scored():
    gt = np.array([[10.0, 0.0], [10.0, 0.0]])
    estimator = np.array([[10.0, 0.0], [10.0, 0.0]])
    active_mask = np.array([False, True])
    mae, n_active = mae_deg_clip(gt, estimator, [], active_mask)
    assert n_active == 1
    assert mae == pytest.approx(10.0, abs=1e-3)


def test_macro_score_invariant_to_ordering_and_sensitive_by_exactly_one_over_n_clips():
    gt = np.array([[10.0, 0.0], [10.0, 0.0]])
    est = np.array([[10.0, 0.0], [10.0, 0.0]])
    mask = np.array([True, True])
    clip_a = ("clip_a", gt, est, [0, 1], mask)
    clip_b = ("clip_b", gt, est, [0, 1], mask)
    score_ab = macro_score_arm([clip_a, clip_b])
    score_ba = macro_score_arm([clip_b, clip_a])
    assert score_ab.macro_mae_deg == pytest.approx(score_ba.macro_mae_deg, abs=1e-9)

    bump = 12.0
    est_bad = np.array([[10.0 + bump, 0.0], [10.0 + bump, 0.0]])
    clip_c = ("clip_c", gt, est_bad, [0, 1], mask)
    two_clip_score = macro_score_arm([clip_a, clip_c])
    assert two_clip_score.macro_mae_deg == pytest.approx((0.0 + bump) / 2.0, abs=1e-2)


def test_macro_and_pooled_diverge_under_unequal_clip_length_pseudoreplication():
    short_gt = np.array([[0.0, 0.0]])
    short_est = np.array([[90.0, 0.0]])
    short_mask = np.array([True])
    long_n = 200
    long_gt = np.tile(np.array([[0.0, 0.0]]), (long_n, 1))
    long_est = np.tile(np.array([[0.0, 0.0]]), (long_n, 1))
    long_mask = np.ones(long_n, dtype=bool)
    clips = [
        ("short_wrong", short_gt, short_est, [0], short_mask),
        ("long_right", long_gt, long_est, list(range(long_n)), long_mask),
    ]
    macro = macro_score_arm(clips)
    pooled = pooled_score_arm(clips)
    assert macro.macro_mae_deg == pytest.approx(90.0 / 2.0, abs=1e-2)  # equal weight: (90 + 0) / 2
    assert pooled.pooled_mae_deg < 1.0  # pseudoreplication: the one bad frame is drowned out by 200 good ones
    assert macro.macro_mae_deg > pooled.pooled_mae_deg  # exactly the divergence clip-macro is meant to catch


def test_clip_level_sign_flip_meet_in_middle_matches_brute_force():
    import itertools

    deltas = [0.5, -0.2, 0.3, 0.1, -0.05, 0.4, -0.15]
    result = exact_sign_flip_over_clips(deltas)
    t_obs = sum(deltas)
    total = 0
    at_least = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(deltas)):
        total += 1
        stat = sum(s * d for s, d in zip(signs, deltas, strict=True))
        if stat >= t_obs - 1e-9:
            at_least += 1
    assert result.permutations == total == 128
    assert result.one_sided_p == pytest.approx(at_least / total, abs=1e-12)
    assert result.min_one_sided_p == pytest.approx(1.0 / 128)


def test_clip_level_sign_flip_at_real_n_test_clips_floor():
    deltas = [0.1] * 21
    result = exact_sign_flip_over_clips(deltas)
    assert result.permutations == 2**21 == 2_097_152
    assert result.min_one_sided_p == pytest.approx(1.0 / 2_097_152, rel=1e-9)
    assert result.one_sided_p == pytest.approx(result.min_one_sided_p, rel=1e-9)  # all-positive is the extreme tail


def test_room_majority_collapse_matches_calibration_note_worked_example():
    rooms = {
        "room10": [1, 1, -1],
        "room15": [1, 1, 1],
        "room16": [1, 1, -1],
        "room2": [1, 1, -1],
        "room23": [1, -1, -1],
        "room24": [1, 1, 1],
        "room8": [1, 1, -1],
    }
    result = room_majority_collapse(rooms)
    assert result.n_rooms == 7
    assert result.n_rooms_favorable == 6
    assert result.one_sided_p == pytest.approx(8.0 / 128.0, abs=1e-12)
    assert result.min_one_sided_p == pytest.approx(1.0 / 128.0, abs=1e-12)


def test_referee_refuses_clip_with_no_active_frame():
    est = np.array([[0.0, 0.0]])
    with pytest.raises(DoaRefereeRefusal):
        mae_deg_clip(np.array([[0.0, 0.0]]), est, [], np.array([False]))




def _flop_model_doa(kind, architecture, total_frames, train_frames):
    runs_gate = kind in (H.ARM_CANDIDATE, H.ARM_RATE_MATCHED_RANDOM)
    infer_per_frame = FLOPS_PER_INFERENCE_ARCH_A if architecture == ARCH_A_ID else FLOPS_PER_INFERENCE_ARCH_B
    train_fn = training_flops_arch_a if architecture == ARCH_A_ID else training_flops_arch_b
    return H.FlopModel(
        featurize_flops=FEATURIZER_FLOPS_PER_FRAME * total_frames,
        gate_infer_flops=infer_per_frame * total_frames if runs_gate else 0,
        downstream_flops_per_firing=FLOPS_PER_REESTIMATE,
        train_flops=train_fn(train_frames, 8) if kind == H.ARM_CANDIDATE else 0,
    )


def _arm_doa(kind, architecture, mae_by_seed, k_by_seed, total_frames, train_frames, seeds):
    params = (3193 if architecture == ARCH_A_ID else 1639) if kind == H.ARM_CANDIDATE else 0
    return H.Arm(
        policy=DOA_BUDGET_POLICY,
        name=f"{kind}@{architecture}",
        kind=kind,
        architecture=architecture,
        total_frames=total_frames,
        params=params,
        flop_model=_flop_model_doa(kind, architecture, total_frames, train_frames),
        seed_results=tuple(
            H.SeedResult(seed=s, metric_value=mae_by_seed[i], actions=k_by_seed[i])
            for i, s in enumerate(seeds)
        ),
    )


def _budget_point_doa(architecture, candidate_mae, rmr_mae, seeds=(0, 1, 2, 3, 4)):
    total_frames = 22569
    train_frames = 25172
    k = [1128] * len(seeds)
    cand = _arm_doa(H.ARM_CANDIDATE, architecture, [candidate_mae] * len(seeds), k, total_frames, train_frames, seeds)
    rmr = _arm_doa(H.ARM_RATE_MATCHED_RANDOM, architecture, [rmr_mae] * len(seeds), k, total_frames, train_frames, seeds)
    ao = _arm_doa(H.ARM_ALWAYS_ON, architecture, [20.0] * len(seeds), [total_frames] * len(seeds), total_frames, train_frames, seeds)
    nu = _arm_doa(H.ARM_NEVER_UPDATE, architecture, [90.0] * len(seeds), [0] * len(seeds), total_frames, train_frames, seeds)
    return H.BudgetPoint(
        DOA_BUDGET_POLICY, "rate_0.05", cand, rmr, ao, nu, architecture=architecture
    )


def test_matched_budget_holds_for_both_architectures():
    point_a = _budget_point_doa(ARCH_A_ID, candidate_mae=10.0, rmr_mae=15.0)
    point_b = _budget_point_doa(ARCH_B_ID, candidate_mae=11.0, rmr_mae=16.0)
    H.assert_matched_ex_training(point_a.candidate, point_a.rate_matched_random)
    H.assert_matched_ex_training(point_b.candidate, point_b.rate_matched_random)
    point_a.certify()
    point_b.certify()
    for arm in (*point_a.arms(), *point_b.arms()):
        assert arm.max_lifecycle_flops() <= FLOP_CEILING

    report = H.run_dual_architecture([point_a], [point_b], wall_ns=1, source_kind="real")
    assert report.candidate_strictly_dominates_rate_matched_random_arch_a is True
    assert report.candidate_strictly_dominates_rate_matched_random_arch_b is True
    assert report.both_architectures_dominate is True
    assert report.verdict == "mechanics-ok"
    assert report.activation_allowed is False
    assert report.scientific_promotion is False
    assert report.independent_scientific_confirmation is False
    assert report.digest() == "03bac31f36c573995aee374be279eb97bdd8dc09d96e89715515e45f5f1ee4d8"


def test_architecture_fragile_fold_when_exactly_one_dominates():
    point_a = _budget_point_doa(ARCH_A_ID, candidate_mae=10.0, rmr_mae=15.0)  # candidate wins
    point_b = _budget_point_doa(ARCH_B_ID, candidate_mae=16.0, rmr_mae=15.0)  # candidate loses (tie-ish, higher)
    report = H.run_dual_architecture([point_a], [point_b], wall_ns=1, source_kind="real")
    assert report.candidate_strictly_dominates_rate_matched_random_arch_a is True
    assert report.candidate_strictly_dominates_rate_matched_random_arch_b is False
    assert report.both_architectures_dominate is False
    assert report.verdict == "architecture-fragile"


def test_null_fold_when_neither_architecture_dominates():
    point_a = _budget_point_doa(ARCH_A_ID, candidate_mae=15.0, rmr_mae=10.0)
    point_b = _budget_point_doa(ARCH_B_ID, candidate_mae=16.0, rmr_mae=15.0)
    report = H.run_dual_architecture([point_a], [point_b], wall_ns=1, source_kind="real")
    assert report.both_architectures_dominate is False
    assert report.verdict == "null"


def test_harness_refuses_uncharged_training_and_k_mismatch_and_ceiling():
    point_a = _budget_point_doa(ARCH_A_ID, candidate_mae=10.0, rmr_mae=15.0)
    cand_no_train = H.Arm(
        policy=DOA_BUDGET_POLICY,
        name="candidate_no_train",
        kind=H.ARM_CANDIDATE,
        architecture=ARCH_A_ID,
        total_frames=point_a.candidate.total_frames,
        params=3193,
        flop_model=H.FlopModel(featurize_flops=1, gate_infer_flops=1, downstream_flops_per_firing=1, train_flops=0),
        seed_results=point_a.candidate.seed_results,
    )
    with pytest.raises(H.UnchargedTraining):
        H.assert_matched_ex_training(cand_no_train, point_a.rate_matched_random)

    huge = H.Arm(
        policy=DOA_BUDGET_POLICY,
        name="candidate_huge",
        kind=H.ARM_CANDIDATE,
        architecture=ARCH_A_ID,
        total_frames=1000,
        params=3193,
        flop_model=H.FlopModel(featurize_flops=FLOP_CEILING, gate_infer_flops=1, downstream_flops_per_firing=1, train_flops=1),
        seed_results=tuple(H.SeedResult(seed=s, metric_value=1.0, actions=1) for s in (0, 1)),
    )
    with pytest.raises(H.CeilingExceeded):
        H.assert_within_ceiling(huge)




def test_sesoi_deg_structural_derivation():
    facts = [
        DoaClipLabelFact(clip_id="c1", n_active_frames=1000, n_changes=20),
        DoaClipLabelFact(clip_id="c2", n_active_frames=500, n_changes=10),
    ]
    sesoi, brackets = compute_doa_sesoi_deg(facts, mean_change_jump_deg=30.0)
    assert sesoi == pytest.approx(0.5625, abs=1e-9)
    assert len(brackets) == 2


def test_sesoi_zero_changes_gives_zero_sesoi_and_prereg_refuses_non_positive():
    facts = [DoaClipLabelFact(clip_id="c1", n_active_frames=1000, n_changes=0)]
    sesoi, _brackets = compute_doa_sesoi_deg(facts, mean_change_jump_deg=10.0)
    assert sesoi == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(DoaPreregRefusal):
        build_doa_prereg(
            timestamp=_TIMESTAMP,
            operating_reestimate_fraction=0.05,
            test_clip_facts=facts,
            test_rooms=["room00"],
            train_change_density=0.05,
            mean_change_jump_deg=10.0,
        )


def test_prereg_seals_before_test_scores_and_carries_promotion_rule():
    facts = [DoaClipLabelFact(clip_id=f"c{i}", n_active_frames=1000, n_changes=20) for i in range(21)]
    body = build_doa_prereg(
        timestamp=_TIMESTAMP,
        operating_reestimate_fraction=0.05,
        test_clip_facts=facts,
        test_rooms=[f"room{i:02d}" for i in range(7)],
        train_change_density=0.05,
        mean_change_jump_deg=30.0,
    )
    assert body["preregistered_before_reading_test_scores"] is True
    assert body["activation_allowed"] is False and body["scientific_promotion"] is False
    assert "canonical_sha256" in body
    assert body["primary_sign_flip_test_plan"]["n_test_clips"] == 21
    assert body["primary_sign_flip_test_plan"]["n_permutations"] == 2**21
    assert body["room_majority_plan"]["n_test_rooms"] == 7
    assert body["architectures"]["both_required"] is True
    assert body["claim_ceiling"]["claim_verb"] == "consistent with"




def test_verifier_imports_stdlib_only():
    source = Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("mop", "numpy", "doa_referee", "doa_producer", "doa_harness", "doa_gate", "doa_labels", "doa_featurizer")
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), name
    allowed_stdlib = {"json", "hashlib", "itertools", "math", "bisect", "dataclasses", "__future__"}
    assert {n.split(".")[0] for n in imported} <= allowed_stdlib




_SMALL_CONFIG = RealDoaBedConfig(seeds=(0, 1, 2), target_rates=(0.10, 0.05), noisy_tv_frames=200, max_frames=600)


@pytest.fixture(scope="module")
def real_doa_artifact(tmp_path_factory):
    if not _REAL_PRESENT:
        pytest.skip("real STARSS23 subset not present")
    out = tmp_path_factory.mktemp("doa_bed")
    return build_real_doa_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=out / "prereg.json")


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_seals_wellformed_artifact_within_ceiling_both_architectures(real_doa_artifact):
    art = real_doa_artifact.artifact
    assert "seal" in art and isinstance(art["seal"], str) and len(art["seal"]) == 64
    assert art["schema"] == "mop-starss23-escs-doa-bed/v1"
    assert art["source_kind"] == "real" and art["rights_clean"] is True
    assert art["verdict"] in ("mechanics-ok", "architecture-fragile", "null")
    assert art["flags"] == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    assert set(art["architectures"]) == {ARCH_A_ID, ARCH_B_ID}
    for architecture in (ARCH_A_ID, ARCH_B_ID):
        for arm in art["harness"]["per_architecture"][architecture]["arm_summaries"]:
            assert arm["max_lifecycle_flops"] <= FLOP_CEILING
        assert art["harness"]["per_architecture"][architecture]["matched_budget"]["flops"] <= FLOP_CEILING


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_receipt_and_seal_byte_reproducible(tmp_path):
    prereg_path = tmp_path / "prereg.json"
    a = build_real_doa_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    b = build_real_doa_bed_artifact(timestamp=_TIMESTAMP, config=_SMALL_CONFIG, prereg_path=prereg_path)
    assert a.artifact["seal"] == b.artifact["seal"]
    assert a.artifact["demonstration_receipt"] == b.artifact["demonstration_receipt"]
    assert a.prereg["canonical_sha256"] == b.prereg["canonical_sha256"]


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_reproduces_referee_and_stats_both_architectures(real_doa_artifact):
    result = V.verify_doa_artifact(real_doa_artifact.artifact)
    assert result.mismatches == ()
    assert result.seal_intact is True
    assert result.scores_reproduced is True
    assert result.stats_reproduced is True
    assert result.honesty_ok is True
    assert result.independent_referee_reproduction is True
    assert result.independent_scientific_confirmation is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_score(real_doa_artifact):
    art = copy.deepcopy(real_doa_artifact.artifact)
    art["per_seed"][ARCH_A_ID][0]["arm_scores_macro"][H.ARM_CANDIDATE]["macro_mae_deg"] += 5.0
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    result = V.verify_doa_artifact(art)
    assert result.seal_intact is True
    assert result.scores_reproduced is False
    assert result.independent_referee_reproduction is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_estimator_track(real_doa_artifact):
    art = copy.deepcopy(real_doa_artifact.artifact)
    some_clip = next(iter(art["corpus_tracks"]))
    for entry in art["corpus_tracks"][some_clip]["estimator_track"]:
        entry[0] = (entry[0] + 45.0) % 360.0
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    assert V.verify_doa_artifact(art).scores_reproduced is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_broken_seal(real_doa_artifact):
    art = copy.deepcopy(real_doa_artifact.artifact)
    art["seal"] = "0" * 64
    result = V.verify_doa_artifact(art)
    assert result.seal_intact is False
    assert result.independent_referee_reproduction is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_never_self_certifies_scientific_confirmation_even_with_reproductions_claim(real_doa_artifact):
    art = real_doa_artifact.artifact
    assert art["reproductions"] == 0
    result = V.verify_doa_artifact(art)
    assert result.independent_scientific_confirmation is False
