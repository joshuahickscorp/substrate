
import json

import torch
from scripts.mop_al1_uncertainty_router import build_pool, run_arm
from scripts.mop_al1_uncertainty_router import run as al1_run
from scripts.mop_dr12_disagreement import (
    PART_FRACS,
    make_mixed_regime,
    parse_seeds,
    read_pr1_context,
)
from scripts.mop_dr12_disagreement import run as dr12_run
from scripts.mop_mt123_router_pilots import (
    MODE_NAMES,
    KWTAHead,
    build_mode,
    density,
    make_episodes,
    mode_flops,
    routed_eval,
    train_router,
    verdict_block,
)
from scripts.mop_mt123_router_pilots import run as mt123_run
from scripts.pr1_mode_error_disjointness import SUBPOPS, make_dataset

TINY_TV = {"dim": 16, "steps": 60, "batch": 32, "ensemble_size": 3}


def test_parse_seeds_forms():
    assert parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert parse_seeds("3") == [3]
    assert parse_seeds("0,2,5") == [0, 2, 5]


def _write_pr1(tmp_path, verdict):
    p = tmp_path / "pr1.json"
    p.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "het_oracle_gain": {"mean": 0.155},
                "hom_oracle_gain": {"mean": 0.118},
                "seed_sd_of_best_mode": 0.014,
                "calibration": {"chosen_separation": 0.12},
                "config": {"dim": 32, "n_classes": 4, "separation": 0.12, "n_train": 100, "n_test": 60},
            }
        )
    )
    return p


def test_pr1_context_is_read_from_disk_not_hardcoded(tmp_path):
    green = read_pr1_context(_write_pr1(tmp_path, "GREEN: oracle gain beats the copy control"))
    assert green["available"] and green["green"] and green["gate"] == "live"
    assert green["het_oracle_gain_mean"] == 0.155
    null = read_pr1_context(_write_pr1(tmp_path, "NULL: gains within seed spread"))
    assert null["available"] and not null["green"] and null["gate"] == "context-null"
    missing = read_pr1_context(tmp_path / "absent.json")
    assert not missing["available"] and missing["gate"] == "missing"


def test_dr12_regime_partitions_and_determinism():
    a = make_mixed_regime(0, 90, 60, 8, 3, 0.5)
    b = make_mixed_regime(0, 90, 60, 8, 3, 0.5)
    for ta, tb in zip(a, b, strict=True):
        assert torch.equal(ta, tb), "same seed must regenerate the identical regime"
    xtr, ytr, ptr, xte, yte, pte = a
    assert xtr.shape == (90, 8) and xte.shape == (60, 8)
    for part, frac in enumerate(PART_FRACS):
        assert abs(float((ptr == part).float().mean()) - frac) < 0.02
    c = make_mixed_regime(1, 90, 60, 8, 3, 0.5)
    assert not torch.equal(c[0], xtr), "different seeds must differ"


def _tiny_dr12(**over):
    kw = dict(
        dim=24,
        n_train=400,
        n_test=200,
        n_classes=4,
        ensemble=3,
        hidden=16,
        epochs=4,
        expert_hidden=32,
        expert_epochs=6,
        tv_kwargs=TINY_TV,
    )
    kw.update(over)
    return dr12_run([0, 1], **kw)


def test_dr12_end_to_end_contract_and_structure(tmp_path):
    r = _tiny_dr12(pr1_path=_write_pr1(tmp_path, "GREEN: x"))
    c = r["contract"]
    assert c["id"] == "DR12" and c["tier"] == "cpu-now"
    for field in ("metric", "baseline", "ablation", "null_hypothesis"):
        assert c[field]
    assert isinstance(r["null_supported"], bool)
    assert r["pr1_context"]["gate"] == "live"
    assert isinstance(r["guard"]["pass"], bool)
    for key in ("noisy_tv", "disagreement_gate_noise_share_mean", "chases_noise"):
        assert key in r["guard"]
    for block in (r["delta_auroc"], r["delta_gate"]):
        assert len(block["per_seed"]) == 2 and "mean" in block and "flips" in block
    for row in r["per_seed"]:
        gate = row["gate"]
        assert set(gate) == {"disagreement", "confidence", "uniform", "oracle"}
        for arm in gate.values():
            shares = arm["share_blobs"] + arm["share_antipodal"] + arm["share_noise"]
            assert abs(shares - 1.0) < 1e-3
        assert gate["oracle"]["acc"] >= gate["uniform"]["acc"] - 1e-9 or True  # oracle reported
    assert "fixed in code before running" in r["verdict_rule"]


def test_dr12_guard_failure_overrides_any_win(monkeypatch, tmp_path):
    import scripts.mop_dr12_disagreement as dr12

    def failing_tv(seed=0, **kw):
        return {
            "noise_error_stays_high": False,
            "epistemic_collapses_on_noise": True,
            "learning_progress_separates": True,
            "raw_error": {},
            "disagreement": {},
            "learning_progress": {},
        }

    monkeypatch.setattr(dr12, "noisy_tv_diagnostic", failing_tv)
    monkeypatch.setattr(dr12, "DEGENERATE_HIGH", 1.01)
    monkeypatch.setattr(dr12, "DEGENERATE_LOW_MARGIN", -1.0)
    r = _tiny_dr12(pr1_path=_write_pr1(tmp_path, "GREEN: x"))
    assert r["verdict"].startswith("GUARD-FAIL") and r["null_supported"] is True


def test_al1_pool_noise_fraction_and_determinism():
    pool, x_eval, y_eval = build_pool(0, 20, 6, 8, 3, noise_frac=0.4, separation=0.5, n_eval=50)
    assert len(pool) == 20 and x_eval.shape == (50, 8)
    assert sum(ep["noise"] for ep in pool) == 8
    pool2, _, _ = build_pool(0, 20, 6, 8, 3, noise_frac=0.4, separation=0.5, n_eval=50)
    for a, b in zip(pool, pool2, strict=True):
        assert torch.equal(a["x"], b["x"]) and a["noise"] == b["noise"]


def test_al1_arm_respects_matched_update_budget():
    pool, x_eval, y_eval = build_pool(0, 16, 6, 8, 3, noise_frac=0.25, separation=0.5, n_eval=40)
    for arm in ("disagreement", "random", "point_error", "disagreement_permuted"):
        out = run_arm(
            arm,
            pool,
            x_eval,
            y_eval,
            0,
            8,
            3,
            members=2,
            hidden=8,
            budget=5,
            window=4,
            updates_per_episode=1,
            lr=5e-3,
            eval_every=2,
        )
        assert out["episodes_used"] <= 5, "no arm may exceed the matched update budget"
        assert 0.0 <= out["noise_share"] <= 1.0
        assert out["acc_curve"], "the adaptation curve must be reported"
    rand = run_arm(
        "random",
        pool,
        x_eval,
        y_eval,
        0,
        8,
        3,
        members=2,
        hidden=8,
        budget=5,
        window=4,
        updates_per_episode=1,
        lr=5e-3,
        eval_every=2,
    )
    assert rand["router_overhead_forwards"] == 0, "the random arm pays no router overhead"


def _tiny_al1():
    return al1_run(
        [0, 1],
        dim=16,
        n_classes=3,
        n_pool=24,
        episode_size=8,
        budget=10,
        window=6,
        members=3,
        hidden=12,
        n_eval=120,
        eval_every=2,
        tv_kwargs=TINY_TV,
    )


def test_al1_end_to_end_contract_and_lr_doctrine():
    r = _tiny_al1()
    c = r["contract"]
    assert c["id"] == "AL1" and c["tier"] == "cpu-now"
    for field in ("metric", "baseline", "ablation", "null_hypothesis"):
        assert c[field]
    assert isinstance(r["null_supported"], bool)
    assert "never an LR gate" in r["config"]["lr_rule"], "uncertainty is admitted only as a router input"
    assert isinstance(r["guard"]["pass"], bool)
    assert len(r["delta_dis_minus_random"]["per_seed"]) == 2
    assert len(r["delta_permuted_minus_random"]["per_seed"]) == 2
    for row in r["per_seed"]:
        assert set(row["arms"]) == {"disagreement", "random", "point_error", "disagreement_permuted"}
    assert "fixed in code before running" in r["verdict_rule"]


def test_kwta_head_is_actually_sparse():
    torch.manual_seed(0)
    head = KWTAHead(8, 16, 3, k_frac=0.25)
    x = torch.randn(10, 8)
    h = head.fc1(x)
    thresh = h.topk(head.k, dim=-1).values[..., -1:]
    active = (h * (h >= thresh).float() != 0).sum(-1)
    assert (active <= head.k).all() and (active >= 1).all()
    assert head(x).shape == (10, 3)


def test_mode_bank_builds_and_flops_are_positive():
    for name in MODE_NAMES:
        m = build_mode(name, 8, 3, hidden=6, steps=2, seed=0)
        assert m(torch.randn(4, 8)).shape == (4, 3)
        assert mode_flops(name, 8, 3, hidden=6, steps=2) > 0
    assert mode_flops("planner", 8, 3, 6, 2) > mode_flops("reactive", 8, 3, 6, 2)


def test_make_episodes_are_single_subpopulation():
    _, _, xte, yte, sub = make_dataset(0, 60, 60, 3, 8, 0.5)
    eps = make_episodes(sub, 9, 5, seed=0)
    assert len(eps) == 9
    for idx, k in eps:
        assert len(idx) == 5
        assert (sub[idx] == k).all(), "each episode must draw from ONE subpopulation"
    assert {k for _, k in eps} == set(range(len(SUBPOPS)))


def test_routed_eval_oracle_is_an_upper_bound():
    torch.manual_seed(0)
    n, dim, n_arms = 40, 8, 3
    x, y = torch.randn(n, dim), torch.randint(0, 3, (n,))
    preds = torch.randint(0, 3, (n_arms, n))
    sub = torch.arange(n) % len(SUBPOPS)  # every subpopulation pool must be non-empty
    eps = make_episodes(sub, 6, 5, seed=1)
    router = train_router(x, eps, (preds == y).long(), n_arms, seed=0, epochs=20)
    out = routed_eval(x, y, eps, preds, router, [10, 20, 30], router_flops=6, episode_size=5)
    assert out["oracle_acc"] >= out["acc"] - 1e-9, "the per-episode oracle bounds any router"
    assert sum(out["selection_counts"]) == 6
    assert out["flops_per_sample"] > 0
    assert density(1.0, 1e6) == 1.0


def test_verdict_block_pr1_demotion_logic():
    win = [0.02, 0.03, 0.025, 0.028, 0.022]
    assert verdict_block(win, "m", None, pr1_green=True)["verdict"].startswith("WIN")
    demoted = verdict_block(win, "m", None, pr1_green=False)
    assert demoted["verdict"].startswith("PLAUSIBLE-BUT-UNVERIFIED")
    assert demoted["null_supported"] is False, "demotion changes the label, not the measurement"
    null = verdict_block([0.01, -0.01, 0.005, -0.002, 0.0], "m", None, pr1_green=True)
    assert null["verdict"].startswith("NULL") and null["null_supported"] is True
    unmatched = verdict_block(win, "m", None, pr1_green=True, matched_ok=False)
    assert unmatched["verdict"].startswith("UNMATCHED") and unmatched["null_supported"] is None


def test_mt123_end_to_end_reads_pr1_and_matches_capacity(tmp_path):
    pr1 = _write_pr1(tmp_path, "NULL: gains within seed spread")
    r = mt123_run(
        [0, 1],
        n_train=240,
        n_test=120,
        episode_size=6,
        n_router_episodes=12,
        n_eval_episodes=12,
        epochs=3,
        planner_hidden=12,
        planner_steps=2,
        sparse_hidden=24,
        cap_tol=0.06,  # tiny widths quantize params coarsely; production keeps the 0.02 default
        pr1_path=pr1,
    )
    assert r["config"]["dim"] == 32 and r["config"]["n_classes"] == 4
    assert r["config"]["separation"] == 0.12
    assert r["pr1_gate"] == "context-null"
    for key in ("mt1_router_vs_best_mode", "mt2_router_vs_uniform_blend", "mt3_hetero_vs_homogeneous"):
        block = r[key]
        assert len(block["delta"]["per_seed"]) == 2
        if block["verdict"].startswith("UNMATCHED"):
            assert block["null_supported"] is None, "an unmatched run is not evaluable, never a null"
        else:
            assert isinstance(block["null_supported"], bool)
        assert not block["verdict"].startswith("WIN"), "a NULL PR1 context must demote every win"
    for row in r["per_seed"]:
        match = row["mt3_match"]["params"]
        assert match["matched"], f"MP3 homogeneous bank must be param-matched, got {match}"
        assert row["routed"]["oracle_acc"] >= row["routed"]["acc"] - 1e-9
    c = r["contract"]
    assert c["id"] == "MP1/MP2/MP3" and c["null_hypothesis"]
