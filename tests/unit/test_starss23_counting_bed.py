from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

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
    _real_noisy_tv_features,
    _run_seed_real,
    build_real_count_bed_artifact,
)
from mop.beds.starss23.count_referee import coast_emitted, mae_clip, score_arm
from mop.beds.starss23.experiments import COUNT_BUDGET_POLICY
from mop.beds.starss23.schema import Clip
from mop.evidence import canonical_sha256
from mop.science import budget as H

FLOP_CEILING = 60_000_000_000
_REAL_PRESENT = DEFAULT_FOA_ROOT.is_dir() and DEFAULT_METADATA_ROOT.is_dir()
_TIMESTAMP = "2026-07-18T00:00:00Z"


def test_count_track_distinct_tracks_and_tail_silence():
    text = "0,1,1,10,0,100\n1,1,1,10,0,100\n1,2,3,20,0,100\n"
    track = count_track_from_metadata_text(text, n_frames=5)
    assert track.tolist() == [1, 2, 0, 0, 0]


def test_count_track_drops_rows_past_end_and_guards_ceiling():
    text = "0,1,1,10,0,100\n9,2,2,20,0,100\n"  # frame 9 is past n_frames
    track = count_track_from_metadata_text(text, n_frames=3)
    assert track.tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
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
    assert clip.n_changes == 3
    assert change_density([clip]) == pytest.approx(3 / 6)
    assert coast_from_zero_mae([clip]) == pytest.approx(4 / 6)


def test_shared_count_seed_lifecycle_matches_the_legacy_projection():
    def clip(clip_id: str, room_id: str, n_frames: int) -> Clip:
        return Clip(clip_id, room_id, n_frames, "0" * 64, ())

    train = (clip("train_a", "r1", 12), clip("train_b", "r2", 11))
    val = (clip("val_a", "r3", 10),)
    test = (clip("test_a", "r4", 13), clip("test_b", "r5", 9))
    all_clips = train + val + test
    rng = np.random.default_rng(123)
    features = {item.clip_id: rng.normal(size=(item.n_frames, 256)) for item in all_clips}
    counts = {item.clip_id: tuple((frame // 3) % 3 for frame in range(item.n_frames)) for item in all_clips}
    estimator = {item.clip_id: np.asarray(counts[item.clip_id], dtype=np.int64) for item in all_clips}
    config = RealCountBedConfig(seeds=(7, 8), target_rates=(0.5, 0.25), noisy_tv_frames=17, epochs=1)
    run = _run_seed_real(
        seed=7,
        train_clips=train,
        val_clips=val,
        test_clips=test,
        features_by_clip=features,
        estimator_by_clip=estimator,
        gt_by_clip=counts,
        noise_features=rng.normal(size=(17, 256)),
        config=config,
        operating_density=0.3,
    )
    assert canonical_sha256(asdict(run)) == (
        "f964d4568daa33f299843018f7aa89b3984f9bf9e327016be5c27af26f9e447d"
    )


def test_shared_marginal_noise_matches_the_legacy_count_bytes():
    noise = _real_noisy_tv_features(7, 5, FrozenCountFeaturizer(), 0.25, 1.5)
    assert hashlib.sha256(noise.tobytes()).hexdigest() == (
        "4f5dbf4c513685a51080ca6c82881ee5e623fea88a1fdc4015461cd4693895a0"
    )
    assert float(noise.mean()) == pytest.approx(0.25)
    assert float(noise.std()) == pytest.approx(1.5)


def test_featurizer_zero_param_and_deterministic():
    fz = FrozenCountFeaturizer()
    assert fz.n_params() == 0
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((4, 2400 * 12))
    f1 = fz.featurize(audio)
    f2 = fz.featurize(audio)
    assert f1.shape == (12, D_CFEAT)
    assert D_CFEAT == 256
    assert np.array_equal(f1, f2)
    assert fz.parameter_digest() == FrozenCountFeaturizer().parameter_digest()
    assert FLOPS_PER_FRAME_COUNT == 1_120_700


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
    assert FLOPS_PER_REESTIMATE == 80_000


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
    assert targets.tolist() == [0, 1, 1, 1, 1, 1]


def test_referee_toy_known_mae():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    reestimates = [1, 3]
    assert coast_emitted(estimator, reestimates) == (0, 1, 1, 2, 2)
    abs_sum, n = mae_clip(gt, estimator, reestimates)
    assert (abs_sum, n) == (2, 5)  # only frame 4 is wrong (emitted 2 vs gt 0)


def test_referee_always_on_equals_mean_abs_e_minus_gt_and_never_update_equals_mean_gt():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    ao_abs, ao_n = mae_clip(gt, estimator, list(range(len(gt))))
    assert ao_abs == sum(abs(estimator[t] - gt[t]) for t in range(len(gt)))
    nu_abs, nu_n = mae_clip(gt, estimator, [])
    assert nu_abs == sum(gt)


def test_referee_pooling_micro_averages_across_clips():
    gt = [0, 1, 1, 2, 0]
    estimator = [0, 1, 2, 2, 0]
    reestimates = [1, 3]
    score = score_arm([(gt, estimator, reestimates), (gt, estimator, reestimates)])
    assert score.abs_error_sum == 4 and score.n_frames == 10
    assert score.mae == pytest.approx(0.4)


def _budget_runs(*, random_action_delta=0):
    rows = []
    for seed in (0, 1, 2):
        per_budget = {}
        for budget_id, shift, actions in (("rate_0.10", 0.0, 100), ("rate_0.05", 0.02, 50)):
            per_budget[budget_id] = {
                "arm_scores": {
                    H.ARM_CANDIDATE: {"mae": 0.30 + shift},
                    H.ARM_RATE_MATCHED_RANDOM: {"mae": 0.34 + shift},
                    H.ARM_ALWAYS_ON: {"mae": 0.25},
                    H.ARM_NEVER_UPDATE: {"mae": 1.25},
                },
                "reestimations": {
                    H.ARM_CANDIDATE: actions,
                    H.ARM_RATE_MATCHED_RANDOM: actions + (random_action_delta if seed == 0 else 0),
                    H.ARM_ALWAYS_ON: 1000,
                    H.ARM_NEVER_UPDATE: 0,
                },
            }
        rows.append(H.BudgetSeedRun(seed, 1000, 1000, 3193, per_budget, "rate_0.05", {}, {}))
    return rows


def _fixture_flops(kind, *, candidate_training=500):
    return H.FlopModel(
        1000,
        100 if kind in (H.ARM_CANDIDATE, H.ARM_RATE_MATCHED_RANDOM) else 0,
        10,
        candidate_training if kind == H.ARM_CANDIDATE else 0,
    )


def _budget_report(runs=None, **options):
    return H.run_matched_budget(
        COUNT_BUDGET_POLICY,
        runs or _budget_runs(),
        score_group="arm_scores",
        score_field="mae",
        action_group="reestimations",
        flop_model=options.pop("flop_model", _fixture_flops),
        operating_budget_id="rate_0.05",
        source_kind="real",
        **options,
    )


def test_harness_payload_is_exact_and_matched():
    report = _budget_report()
    assert report.candidate_strictly_dominates_rate_matched_random is True
    assert report.verdict == "mechanics-ok"
    assert (
        canonical_sha256(report.payload())
        == "0b489f0e2304ce5e458d68bad6ac0243f0b669d424f7d67b7f08287be7443fdf"
    )


def test_harness_refuses_uncharged_training_and_action_mismatch():
    with pytest.raises(H.BudgetRefusal, match="training cost"):
        _budget_report(flop_model=lambda kind: _fixture_flops(kind, candidate_training=0))
    with pytest.raises(H.BudgetRefusal, match="action count"):
        _budget_report(_budget_runs(random_action_delta=1))


def test_harness_refuses_ceiling_exceeded():
    with pytest.raises(H.BudgetRefusal, match="ceiling"):
        _budget_report(ceiling=2000)


def test_pareto_minimizes_flops_and_mae():
    pareto = {(row["budget_id"], row["arm_kind"]) for row in _budget_report().payload()["pareto"]}
    assert ("rate_0.10", H.ARM_RATE_MATCHED_RANDOM) not in pareto
    assert ("rate_0.05", H.ARM_RATE_MATCHED_RANDOM) in pareto


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
    assert round(0.02 / cb["per_frame_granularity"]) == 451
    assert cb["one_clip_change_mass_mae"] == pytest.approx(0.0238, abs=5e-4)
    assert body["sign_flip_test_plan"]["min_one_sided_p"] == pytest.approx(1 / 32)
    assert body["sign_flip_test_plan"]["two_sided_alpha_reachable"] is False


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
    assert set(n.split(".")[0] for n in imported) <= {
        "json",
        "hashlib",
        "itertools",
        "dataclasses",
        "__future__",
    }


_SMALL_CONFIG = RealCountBedConfig(
    seeds=(0, 1, 2), target_rates=(0.10, 0.05), noisy_tv_frames=400, max_frames=100
)


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
    for arm in art["harness"]["arm_summaries"]:
        assert arm["max_lifecycle_flops"] <= FLOP_CEILING
    assert art["matched_budget"]["flops"] <= FLOP_CEILING


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_producer_receipt_and_seal_byte_reproducible(tmp_path):
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
    assert result.independent_scientific_confirmation is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_score(real_count_artifact):
    art = copy.deepcopy(real_count_artifact.artifact)
    art["per_seed"][0]["arm_scores"]["candidate"]["mae"] += 0.5
    body = {k: v for k, v in art.items() if k != "seal"}
    art["seal"] = V._canonical_sha256(body)
    result = V.verify_count_artifact(art)
    assert result.seal_intact is True
    assert result.scores_reproduced is False
    assert result.independent_referee_reproduction is False


@pytest.mark.skipif(not _REAL_PRESENT, reason="real STARSS23 subset not present")
def test_verifier_detects_tampered_reestimates_and_estimator_track(real_count_artifact):
    art = copy.deepcopy(real_count_artifact.artifact)
    clip0 = art["per_seed"][0]["clips"][0]
    if clip0["reestimate_frames"]["candidate"]:
        clip0["reestimate_frames"]["candidate"] = clip0["reestimate_frames"]["candidate"][:-1]
    art["seal"] = V._canonical_sha256({k: v for k, v in art.items() if k != "seal"})
    assert V.verify_count_artifact(art).scores_reproduced is False

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
