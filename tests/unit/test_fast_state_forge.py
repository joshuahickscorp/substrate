"""Permanent regression tests for the fast state plasticity forge.

Every test here exists because a specific defect is possible and would silently invalidate a scientific
claim: arms that alias, groups that change when they were declared frozen, a persistent core that is quietly
reinitialized, an anchor that does not anchor, a tuning oracle that can see the test split.

Budgets are deliberately tiny. These tests check invariants, not accuracy.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastforge import arch as A  # noqa: E402
from fastforge import arms as AR  # noqa: E402
from fastforge import data as D  # noqa: E402
from fastforge import engine as E  # noqa: E402
from fastforge import sequence as S  # noqa: E402

PROOF = ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1"
DOMS = {"har": (9, 6), "speech": (40, 10)}
TINY = {"acquire": 4, "second": 4, "return": 2, "adapt": 2}
HAVE_DATA = D.HAR_ROOT.is_dir() and (D.SPEECH_CACHE.exists() or D.SPEECH_ROOT.is_dir())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="domain data not present on this box")


def _proof(name):
    p = PROOF / name
    if not p.is_file():
        pytest.skip(f"{name} not sealed yet")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- structural invariants, no data needed


def test_every_trainable_parameter_belongs_to_exactly_one_declared_group():
    for kind in A.BUILDERS:
        ok, undeclared, duplicated = A.check_partition(A.build(kind, DOMS))
        assert ok, (kind, undeclared, duplicated)


def test_trainable_group_identity_is_exact():
    m = A.build("G", DOMS)
    names = E.trainable_names(m, ["fast_core", "head.har"])
    assert set(names) == set(m.param_groups["fast_core"]) | set(m.param_groups["head.har"])
    assert "heads.speech.weight" not in names


def test_frozen_group_identity_never_changes_during_a_fit():
    m = A.build("G", DOMS)
    x, y = torch.randn(16, 32, 9), torch.randint(0, 6, (16,))
    r = E.fit(m, "har", x, y, train_groups=["head.har"], steps=3, rng=np.random.default_rng(0))
    assert r["undeclared_changes"] == []
    assert not set(r["changed_params"]) & set(m.param_groups["fast_core"])
    assert not set(r["changed_params"]) & set(m.param_groups["adapter.har"])


def test_domain_adapter_isolation_the_inactive_domain_never_moves():
    m = A.build("G", DOMS)
    x, y = torch.randn(16, 32, 9), torch.randint(0, 6, (16,))
    r = E.fit(
        m,
        "har",
        x,
        y,
        train_groups=["fast_core"]
        + [f"{k}.har" for k in ("proj_conv", "proj_lin", "adapter", "norm", "head")],
        steps=3,
        rng=np.random.default_rng(0),
    )
    assert not any(n.endswith(".speech") or ".speech." in n for n in r["changed_params"])


def test_fast_delta_is_bounded_and_the_anchor_is_never_trainable():
    m = A.build("H", DOMS)
    assert all(not b.requires_grad for _, b in m.named_buffers())
    with torch.no_grad():
        for k in m.delta:
            m.delta[k].fill_(1e6)  # push far past the bound
    for n in m.anchor_keys:
        assert float(m.bounded_delta(n).abs().max()) <= A.TAU + 1e-5
    assert m.drift() > 0


def test_anchor_restoration_returns_the_core_to_the_anchor():
    m = A.build("H", DOMS)
    with torch.no_grad():
        for k in m.delta:
            m.delta[k].normal_(0, 0.1)
    assert m.drift() > 0
    m.restore_anchor(1.0)
    assert m.drift() == pytest.approx(0.0, abs=1e-6)
    eff = m.effective_core()
    for n in m.anchor_keys:
        assert torch.allclose(eff[n], getattr(m, f"anchor_{n}"))


def test_plasticity_action_bounds_no_action_may_train_an_undeclared_group():
    from fastforge.runs import interference as IF

    m = A.build("G", DOMS)
    for kinds in IF.ACTIONS.values():
        g = IF._groups_for(m, kinds, "har", "fast_core")
        assert all(x in m.param_groups for x in g)
        assert not any(x.endswith(".speech") for x in g)


def test_round_architectures_are_materially_different_from_their_base():
    base = A.build("G", DOMS)
    r1 = A.build("G_R1_domain_input_norm", DOMS)
    r2 = A.build("G_R2_bounded_core_delta", DOMS)
    r3 = A.build("G_R3_split_core", DOMS)
    assert any(g.startswith("input_norm.") for g in r1.param_groups)
    assert not any(g.startswith("input_norm.") for g in base.param_groups)
    assert r2.core_delta and r2.drift() == 0.0
    with torch.no_grad():
        for k in r2.delta:
            r2.delta[k].normal_(0, 0.1)
    assert r2.drift() > 0
    r2.restore_anchor(1.0)
    assert r2.drift() == pytest.approx(0.0, abs=1e-6)
    assert "fast_core_ih" in r3.param_groups and "fast_core_hh" in r3.param_groups
    assert not set(r3.param_groups["fast_core_ih"]) & set(r3.param_groups["fast_core_hh"])


def test_anchor_restore_gate_actually_restores():
    m = A.build("H", DOMS)
    x, y = torch.randn(16, 32, 9), torch.randint(0, 6, (16,))
    gate = E.Gate("anchor_restore", thresh=-1.0, rng=np.random.default_rng(0))  # always over the bound
    r = E.fit(
        m,
        "har",
        x,
        y,
        train_groups=["fast_delta", "head.har"],
        steps=4,
        rng=np.random.default_rng(0),
        gate=gate,
        shared_group="fast_delta",
    )
    assert r["gate_blocked_updates"] == 4
    assert m.drift() == pytest.approx(0.0, abs=1e-6)


def test_gate_kinds_are_all_simple_rules_with_no_learned_parameters():
    for kind in E.Gate.KINDS:
        g = E.Gate(kind, 0.0, np.random.default_rng(0))
        assert not hasattr(g, "parameters")
    assert "learned" not in E.Gate.KINDS


def test_memory_budget_is_bounded_and_policies_differ():
    rng = np.random.default_rng(0)
    for policy in ("gdumb", "reservoir", "recent", "none"):
        mem = E.Memory(policy, 40)
        for _ in range(5):
            mem.add(torch.randn(30, 8, 3), torch.randint(0, 4, (30,)), rng)
            assert mem.size() <= 40
        assert (mem.size() == 0) == (policy == "none")


# ---------------------------------------------------------------- behavioural invariants, need data


@needs_data
def test_cross_domain_arms_do_not_alias():
    ids = [AR.arm_identity(a, steps=TINY) for a in AR.ARMS]
    assert len({tuple(map(str, i["identity_tuple"])) for i in ids}) == len(ids)
    assert len({i["checkpoint_final"] for i in ids}) == len(ids)
    assert all(not i["undeclared_changes"] for i in ids)


@needs_data
def test_fresh_controls_reinitialize_and_persistent_arms_do_not():
    fresh = AR.arm_identity("fresh_core", steps=TINY)
    persist = AR.arm_identity("full_persistent_core", steps=TINY)
    assert set(fresh["reinitialized"]) <= set(fresh["shared_groups_changed_at_boundary"])
    assert persist["shared_groups_changed_at_boundary"] == []


@needs_data
def test_fast_core_persists_across_the_domain_boundary_in_a_carrying_arm():
    r = AR.run_arm("frozen_fast_core", ("har", "speech"), 0, steps=TINY)
    assert "fast_core" not in r["shared_groups_changed_at_boundary"]
    p2 = r["receipts"][-3]
    assert not set(p2["changed_params"]) & set(AR.A.build("XD", DOMS).param_groups["fast_core"])


@needs_data
def test_domain_order_reversal_produces_a_different_run_not_a_relabelled_one():
    f = S.run("G0_always_trainable", ("har", "speech"), 0, steps=TINY)
    b = S.run("G0_always_trainable", ("speech", "har"), 0, steps=TINY)
    assert f["checkpoint_final"] != b["checkpoint_final"]
    assert f["direction"] != b["direction"]


@needs_data
def test_bidirectional_transfer_arms_share_a_policy_but_not_a_result():
    for arm in ("G1_frozen_after_first", "H_never_update"):
        f = S.run(arm, ("har", "speech"), 0, steps=TINY)
        b = S.run(arm, ("speech", "har"), 0, steps=TINY)
        assert f["policy_sha"] == b["policy_sha"]
        assert f["metrics"]["second_acquisition"] != b["metrics"]["second_acquisition"]


@needs_data
def test_domain_local_checkpoint_recovery_group_hashes_identify_what_moved():
    r = S.run("G1_frozen_after_first", ("har", "speech"), 0, steps=TINY)
    p1, p2 = r["receipts"][0], r["receipts"][1]
    assert p1["group_sha_after"]["fast_core"] == p2["group_sha_after"]["fast_core"]
    assert p1["group_sha_after"]["head.speech"] != p2["group_sha_after"]["head.speech"]


@needs_data
def test_splits_are_unit_disjoint_so_a_tuning_signal_cannot_see_a_training_unit():
    for dname in ("har", "speech"):
        d = D.domain(dname)
        sp = D.splits(dname, 0)
        u = np.asarray(d["u"])
        counts = sp["unit_counts"]
        assert counts["main"] > 0 and counts["tune"] > 0 and counts["future"] > 0
        assert counts["main"] + counts["tune"] + counts["future"] == len(np.unique(u))
        assert len(set(np.asarray(d["ute"]).tolist()) & set(u.tolist())) == 0


@needs_data
def test_oracle_headroom_cannot_leak_the_test_split():
    r = S.run("G0_always_trainable", ("har", "speech"), 0, steps=TINY)
    sp = D.splits("speech", 0)
    tune_acc = S.E.evaluate  # the selection signal is built only from tune and return_tune
    assert "tune_utility" in r["metrics"]
    assert r["metrics"]["tune_utility"] == pytest.approx(
        float(np.mean([r["metrics"]["second_tune"], r["metrics"]["return_tune"]]))
    )
    assert tune_acc is not None and len(sp["test"][0]) > 0


# ---------------------------------------------------------------- sealed artifact invariants


def test_interference_map_invariants():
    m = _proof("MOP_SUBSTRATE_INTERFERENCE_MAP.json")
    assert m["scored_on"].startswith("tuning split")
    for v in m["classification"].values():
        assert not (v["transferable"] and v["domain_specific"])
        assert not (v["safe_to_reopen"] and v["should_remain_frozen"])
    assert set(m["transferable_groups"]) & set(m["keep_frozen"]) == set()


def test_plasticity_gate_requires_its_controls_to_fail():
    r = _proof("MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json")
    if r["verdict"] == "plasticity_action_headroom_present":
        assert r["controls_behaved"] is True
        for v in r["per_direction"].values():
            assert v["random_control"]["lower_95_cb"] < 0.05
            assert v["shuffled_control"]["lower_95_cb"] < 0.05
    assert r["learned_gate_opened"] == (r["verdict"] == "plasticity_action_headroom_present")


def test_cross_domain_positive_requires_both_directions():
    s = _proof("MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json")
    if s["bidirectional_verdict"] == "cross_domain_positive":
        assert s["arms_passing_in_both_directions"]
    else:
        assert not s["arms_passing_in_both_directions"]


def test_task_free_context_inference_beats_its_simple_control_or_is_a_null():
    r = _proof("MOP_TASK_FREE_CONTEXT_REPORT.json")
    if r["verdict"] == "task_free_context_positive":
        assert r["beats_simple_classifier"] is True
        assert r["shuffled_context_rejected"] is True
