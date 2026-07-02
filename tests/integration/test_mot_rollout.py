"""WP-05 rollout/planning harness tests: tiny synthetic tensors only, seconds per test, no encoder
loads, no network. Each DR6/DR11/DR13 script runs end to end at toy scale through its Experiment
contract, and the honesty machinery (matched FLOPs, preregistered nulls, true-env scoring, guards)
is asserted structurally rather than on the scientific verdicts (toy scale is not evidence)."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import mop_dr6_rollout_planning as dr6  # noqa: E402
import mop_dr11_mc_rollouts as dr11  # noqa: E402
import mop_dr13_horizon_limit as dr13  # noqa: E402

from mop.devices import resolve  # noqa: E402

DEV = resolve("cpu")


_TINY_COMMON = {
    "seeds": [0, 1],
    "dim": 8,
    "action_dim": 2,
    "hidden": 16,
    "epochs": 25,
    "lr": 1e-3,
    "n_train_transitions": 200,
    "nonlinearity": 0.5,
}


def _tiny_dr6_cfg():
    over = dict(_TINY_COMMON, noise=0.02, n_trials=4, horizon=3, n_samples=12, cem_iters=2, rollout_k=3)
    return OmegaConf.merge(dr6.default_cfg(), OmegaConf.create(over))


def _tiny_dr11_cfg():
    over = dict(
        _TINY_COMMON,
        noise_learnable=0.3,
        noise_dominant=3.0,
        n_trials=3,
        horizon=3,
        n_cand=6,
        k_mc=3,
        n_exec=4,
    )
    return OmegaConf.merge(dr11.default_cfg(), OmegaConf.create(over))


def _tiny_dr13_cfg():
    over = dict(_TINY_COMMON, noise=0.05, horizons=[1, 2], n_trials=3, n_samples=10)
    return OmegaConf.merge(dr13.default_cfg(), OmegaConf.create(over))


# ---------------------------------------------------------------- shared helpers


def test_parse_seeds_forms():
    for mod in (dr6, dr11, dr13):
        assert mod.parse_seeds("0-4") == [0, 1, 2, 3, 4]
        assert mod.parse_seeds("0,2,5") == [0, 2, 5]
        assert mod.parse_seeds("3") == [3]


def test_contracts_declared():
    for cls in (dr6.DR6RolloutPlanning, dr11.DR11MCRollouts, dr13.DR13HorizonLimit):
        c = cls().contract()
        assert c["null_hypothesis"] and c["baseline"] and c["ablation"]
        assert c["tier"] == "cpu-now"


# ---------------------------------------------------------------- DR6


def test_dr6_runs_and_reports_contract(tmp_path):
    out = dr6.DR6RolloutPlanning().run(_tiny_dr6_cfg(), DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert out["survives"] == (not out["null_supported"])
    assert len(out["per_seed"]) == 2
    assert (tmp_path / "dr6_rollout_planning.json").exists()
    # preregistered thresholds are in the artifact, not derived from results
    assert out["preregistered"]["planning_margin"] == dr6.PLANNING_MARGIN
    assert out["preregistered"]["r2_floor"] == dr6.R2_FLOOR


def test_dr6_planner_and_greedy_are_flop_matched(tmp_path):
    out = dr6.DR6RolloutPlanning().run(_tiny_dr6_cfg(), DEV, tmp_path)
    for r in out["per_seed"]:
        comp = r["compute"]
        assert comp["planner_forwards_per_trial"] == comp["greedy_forwards_per_trial"]
        assert comp["planner_vs_greedy_matched"]["matched"] is True
        # the shuffle control never spends MORE than the planner
        assert comp["shuffle_forwards_per_trial"] <= comp["planner_forwards_per_trial"]
        assert "rollout_gate" in r and "planning_licensed" in r["rollout_gate"]


def test_dr6_true_env_execution_is_deterministic_given_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    from mop.experiments.ex2_latent_planning import _true_dynamics_params

    params = _true_dynamics_params(8, 2, 0.5, torch.Generator().manual_seed(0))
    seq = torch.randn(3, 2)
    z0, goal = torch.randn(8), torch.randn(8)
    d1 = dr6.execute_true(seq, z0, goal, params, 0.5, g1)
    d2 = dr6.execute_true(seq, z0, goal, params, 0.5, g2)
    assert d1 == d2  # noise=0 execution: the honest yardstick is reproducible


# ---------------------------------------------------------------- DR11


def test_dr11_runs_with_guard_and_matched_flops(tmp_path):
    out = dr11.DR11MCRollouts().run(_tiny_dr11_cfg(), DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert (tmp_path / "dr11_mc_rollouts.json").exists()
    for r in out["per_seed"]:
        lc = r["learnable_regime"]["compute"]
        assert lc["mc_forwards_per_trial"] == lc["det_forwards_per_trial"]
        assert lc["matched"]["matched"] is True
        # both regimes present: the noisy-TV guard arm is not optional
        assert "noise_dominant_regime" in r
        assert isinstance(r["noisy_tv_guard_ok"], bool)
        assert r["learnable_regime"]["diversity"] >= 0.0


def test_dr11_stochastic_rollout_actually_varies():
    from mop.experiments.ex2_latent_planning import _DynamicsModel

    model = _DynamicsModel(8, 2, 16)
    z0 = torch.randn(8)
    seqs = torch.randn(4, 3, 2)
    g = torch.Generator().manual_seed(0)
    t1 = dr11.stochastic_rollout(model, z0, seqs, resid_std=0.5, g=g)
    t2 = dr11.stochastic_rollout(model, z0, seqs, resid_std=0.5, g=g)
    assert not torch.allclose(t1, t2)  # samples differ: the diversity statistic can be nonzero
    d1 = dr11.deterministic_rollout(model, z0, seqs)
    d2 = dr11.deterministic_rollout(model, z0, seqs)
    assert torch.allclose(d1, d2)  # the matched arm is deterministic by construction


# ---------------------------------------------------------------- DR13


def test_dr13_runs_horizon_sweep(tmp_path):
    out = dr13.DR13HorizonLimit().run(_tiny_dr13_cfg(), DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert (tmp_path / "dr13_horizon_limit.json").exists()
    for r in out["per_seed"]:
        assert set(r["per_horizon"].keys()) == {"1", "2"}
        for h in ("1", "2"):
            comp = r["per_horizon"][h]["compute"]
            assert comp["planner_forwards"] == comp["reactive_forwards"]
            assert comp["matched"]["matched"] is True
        assert set(r["kstep_r2_decay"].keys()) == {"1", "2"}
        assert r["crossover_horizon"] in (0, 1, 2)
        # a crossover requires a win AND a later loss, never just a win
        if r["crossover_observed"]:
            assert r["winning_horizons"] and r["crossover_horizon"] < 2


def test_dr13_null_reads_only_nontrivial_wins(tmp_path):
    cfg = _tiny_dr13_cfg()
    cfg.seeds = [0]
    # structural check on the aggregation rule: survives demands a win at some horizon > 1
    out = dr13.DR13HorizonLimit().run(cfg, DEV, tmp_path)
    r = out["per_seed"][0]
    assert out["survives"] == (r["nontrivial_win"] and r["model_licensed"])
