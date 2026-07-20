
import json
import sys
from pathlib import Path

import torch

from mop.devices import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mop_dr9_verify_revise as dr9  # noqa: E402
import mop_mt7_beam_search as mt7  # noqa: E402
import mop_mt8_latent_debate as mt8  # noqa: E402

DEV = resolve("cpu")

TINY = dict(dim=16, hidden=32, verifier_hidden=16, n_classes=4, samples=240, epochs=30, lr=1e-2)


def tiny_backbone(seed=0, n_max=3):
    x, y, _ = mt7.make_ambiguous_split(200, TINY["dim"], TINY["n_classes"], 3.0, 0.5, seed)
    refiner, head, verifier = dr9.train_backbone(
        x, y, TINY["n_classes"], TINY["dim"], TINY["hidden"], TINY["verifier_hidden"], n_max, 20, 1e-2, seed
    )
    return refiner, head, verifier, x, y


def test_mt7_flop_schedule_counts_pruned_work():
    sched = mt7.beam_flop_schedule(rounds=3, beam_width=3, expansions=2, per_step=100, per_eval=10)
    assert sched["expansions"] == 12
    assert sched["total_flops"] == 12 * 110
    assert sched["kept_widths"] == [2, 3, 3]


def test_mt7_beam_k1_e1_reduces_to_greedy():
    refiner, head, verifier, x, y = tiny_backbone(seed=0, n_max=3)
    beam = mt7.beam_search_eval(refiner, head, verifier, x, y, 1, 1, 3, 0.7, "trained", seed=0)
    greedy = mt7.greedy_eval(refiner, head, x, y, 3)
    assert beam == greedy  # a width-1 beam with no noise children IS the greedy chain


def test_mt7_runs_and_matches_total_flops(tmp_path):
    cfg = mt7.default_cfg([0, 1], beam_width=3, expansions=2, rounds=3, **TINY)
    out = mt7.MT7BeamSearch().run(cfg, DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert len(out["per_seed"]) == 2
    assert out["greedy_matched_depth"] >= 3
    assert set(out["compute"]) >= {"matched", "ratio", "tol"}
    assert out["compute"]["matched"] is True  # matched by construction of the greedy depth
    for rec in out["per_seed"]:
        for key in (
            "beam_acc",
            "greedy_matched_acc",
            "greedy_horizon_acc_unmatched",
            "shuffled_scorer_acc",
            "oracle_beam_acc",
        ):
            assert 0.0 <= rec[key] <= 1.0
        assert isinstance(rec["oracle_headroom_over_greedy"], bool)
        assert "regime_calibrated" in rec["d3"]
    assert out["preregistered"]["flop_tol"] == mt7.FLOP_TOL
    assert isinstance(out["verdict"], str) and out["verdict"]


def test_mt8_zero_init_wires_start_at_the_independent_module():
    refiner, head, verifier, x, y = tiny_backbone(seed=1, n_max=3)
    mod = mt8.ModuleShell(refiner, head, verifier).freeze()
    wires = mt8.DebateWires(TINY["dim"])  # msg wires zero-initialized, referee untrained
    with torch.no_grad():
        logits, info = mt8.debate_forward(mod, mod, wires, x, 3)
        solo = mt8.unroll_logits(mod, x, 3)
    assert info["dist_mean"] == 0.0
    assert torch.allclose(logits, solo, atol=1e-5)


def test_mt8_pr1_gate_reader_paths(tmp_path):
    missing = mt8.read_pr1_context(tmp_path / "absent.json")
    assert missing == {
        "available": False,
        "green": False,
        "gate": "missing",
        "note": missing["note"],
    }
    green_p = tmp_path / "green.json"
    green_p.write_text(json.dumps({"verdict": "GREEN: het oracle gain clears hom + SD", "config": {}}))
    live = mt8.read_pr1_context(green_p)
    assert live["green"] is True and live["gate"] == "live"
    null_p = tmp_path / "null.json"
    null_p.write_text(json.dumps({"verdict": "NULL: gains within seed spread", "config": {}}))
    ctx = mt8.read_pr1_context(null_p)
    assert ctx["green"] is False and ctx["gate"] == "context-null"


def test_mt8_runs_all_arms_and_reports_gate(tmp_path):
    pr1_p = tmp_path / "pr1.json"
    pr1_p.write_text(json.dumps({"verdict": "GREEN: fixture", "config": {}}))
    cfg = mt8.default_cfg([0], rounds=2, pr1_path=str(pr1_p), **TINY)
    out = mt8.MT8LatentDebate().run(cfg, DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert out["pr1"]["gate"] == "live"
    rec = out["per_seed"][0]
    for key in (
        "debate_acc",
        "best_single_acc",
        "ensemble_acc",
        "self_debate_acc",
        "shuffled_partner_acc",
    ):
        assert 0.0 <= rec[key] <= 1.0
    assert -1.0 <= rec["module_error_correlation"] <= 1.0
    assert 0.0 <= rec["referee_weight_mean"] <= 1.0
    b = out["flop_budget"]
    assert b["n_max_train"] >= max(b["steps_single"], b["steps_ensemble"], 2)
    assert out["compute_vs_single"]["matched"] is True
    assert out["compute_vs_ensemble"]["matched"] is True
    assert out["claim_strength"] in {"verified", "plausible_unverified", "n/a"}
    assert isinstance(out["verdict"], str) and out["verdict"]


def test_mt8_missing_pr1_still_runs_as_context(tmp_path):
    cfg = mt8.default_cfg([0], rounds=2, pr1_path=str(tmp_path / "nope.json"), **TINY)
    out = mt8.MT8LatentDebate().run(cfg, DEV, tmp_path)
    assert out["pr1"]["gate"] == "missing"
    assert isinstance(out["null_supported"], bool)  # the row runs and reports against context


def test_wp04_experiments_declare_contract():
    for cls in (mt7.MT7BeamSearch, mt8.MT8LatentDebate):
        exp = cls()
        assert exp.id and exp.metric and exp.baseline and exp.ablation
        assert exp.null_hypothesis and exp.tier == "cpu-now"
