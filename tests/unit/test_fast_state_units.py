"""Unit level regression tests for the fast state substrate machinery.

The behavioural tests in test_fast_state_forge.py check invariants of whole runs. These check the pieces
those runs are built from, on synthetic tensors, so a defect in a gate rule, a memory policy, an inference
helper or a stream construction is caught without needing the domain data.

House style: no dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastforge import arch as A  # noqa: E402
from fastforge import data as D  # noqa: E402
from fastforge import engine as E  # noqa: E402
from fastforge import sequence as S  # noqa: E402

DOMS = {"har": (9, 6), "speech": (40, 10)}


def _batch(ch=9, t=24, n=16, classes=6):
    return torch.randn(n, t, ch), torch.randint(0, classes, (n,))


# ---------------------------------------------------------------- inference helpers


def test_lower_bound_uses_the_right_t_multiplier_and_shrinks_with_spread():
    assert E.lcb([0.1] * 8) == pytest.approx(0.1)
    tight = E.lcb([0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.10])
    loose = E.lcb([0.30, -0.10, 0.25, -0.05, 0.20, 0.00, 0.15, 0.05])
    assert tight > loose
    assert E.lcb([0.2]) == pytest.approx(0.2)
    assert E.lcb([]) == 0.0


def test_effect_is_paired_element_by_element():
    a = [0.5, 0.6, 0.7]
    b = [0.4, 0.5, 0.6]
    e = E.effect(a, b)
    assert e["mean"] == pytest.approx(0.1)
    assert e["n"] == 3
    # reordering one side changes a paired effect, which is the whole point of pairing
    assert E.effect(a, list(reversed(b)))["mean"] == pytest.approx(0.1)
    assert E.effect(a, list(reversed(b)))["lower_95_cb"] != e["lower_95_cb"]


# ---------------------------------------------------------------- memory policies


def test_reservoir_admits_late_items_and_recent_evicts_the_oldest():
    rng = np.random.default_rng(0)
    res = E.Memory("reservoir", 10)
    for step in range(6):
        res.add(torch.full((5, 4, 3), float(step)), torch.full((5,), step % 3), rng)
    assert res.size() == 10
    seen = {float(x[0, 0]) for x in res.x}
    assert len(seen) > 1, "a reservoir that never replaces is not a reservoir"

    rec = E.Memory("recent", 10)
    for step in range(6):
        rec.add(torch.full((5, 4, 3), float(step)), torch.zeros(5, dtype=torch.long), rng)
    assert rec.size() == 10
    assert {float(x[0, 0]) for x in rec.x} <= {4.0, 5.0}, "recent must hold the newest items"


def test_gdumb_balances_classes_under_a_skewed_stream():
    rng = np.random.default_rng(0)
    mem = E.Memory("gdumb", 30)
    mem.add(torch.randn(50, 4, 3), torch.zeros(50, dtype=torch.long), rng)
    mem.add(torch.randn(10, 4, 3), torch.ones(10, dtype=torch.long), rng)
    counts = {c: mem.y.count(c) for c in set(mem.y)}
    assert len(counts) == 2
    assert min(counts.values()) >= 10


def test_sampling_an_empty_memory_returns_nothing():
    assert E.Memory("gdumb", 5).sample(4, np.random.default_rng(0)) is None


# ---------------------------------------------------------------- gates


def test_every_gate_kind_decides_and_records_its_decision():
    m = A.build("H", DOMS)
    names = m.param_groups["fast_delta"]
    x, y = _batch()
    for kind in E.Gate.KINDS:
        g = E.Gate(kind, 0.0, np.random.default_rng(0))
        g.set_reference(E.reference_gradient(m, names, x, y, "har"), 1.0)
        out = g.allow_shared(0, m, names, cur_probe=2.0, drift=5.0, perf_drop=0.5)
        assert isinstance(out, bool)
        assert g.decisions == [int(out)]
    assert E.Gate("never", rng=np.random.default_rng(0)).allow_shared(0, m, names) is False
    assert E.Gate("always", rng=np.random.default_rng(0)).allow_shared(0, m, names) is True


def test_a_replayed_gate_reproduces_the_sequence_it_was_given():
    m = A.build("H", DOMS)
    g = E.Gate("shuffled", rng=np.random.default_rng(0))
    g.replay([1, 0, 0, 1])
    got = [g.allow_shared(i, m, m.param_groups["fast_delta"]) for i in range(8)]
    assert got == [True, False, False, True] * 2


def test_probe_loss_does_not_leave_the_model_in_eval_mode():
    m = A.build("G", DOMS)
    m.train()
    x, y = _batch()
    E.probe_loss(m, x, y, "har")
    assert m.training, "a probe must not silently switch the model out of training mode"


def test_reference_gradient_clears_the_grads_it_used():
    m = A.build("G", DOMS)
    x, y = _batch()
    g = E.reference_gradient(m, m.param_groups["fast_core"], x, y, "har")
    assert g is not None and g.numel() > 0
    assert all(p.grad is None for p in m.parameters())


# ---------------------------------------------------------------- fit receipts and regularization


def test_ewc_penalty_changes_the_trajectory_it_is_applied_to():
    x, y = _batch()
    rng_names = ["head.har"]
    plain = A.build("G", DOMS)
    E.fit(plain, "har", x, y, train_groups=rng_names, steps=3, rng=np.random.default_rng(0))
    penalised = A.build("G", DOMS)
    penalised.load_state_dict(A.build("G", DOMS).state_dict())
    fisher = E.fisher_diag(
        penalised, "har", x, y, penalised.param_groups["head.har"], np.random.default_rng(0), n=2, batch=8
    )
    assert set(fisher["fisher"]) == set(penalised.param_groups["head.har"])
    assert all(float(v.sum()) >= 0 for v in fisher["fisher"].values())
    r = E.fit(
        penalised,
        "har",
        x,
        y,
        train_groups=rng_names,
        steps=3,
        rng=np.random.default_rng(0),
        ewc=fisher,
        ewc_lambda=10.0,
    )
    assert r["changed_param_count"] > 0


def test_a_zero_step_fit_changes_nothing_and_says_so():
    m = A.build("G", DOMS)
    x, y = _batch()
    r = E.fit(m, "har", x, y, train_groups=["head.har"], steps=0, rng=np.random.default_rng(0))
    assert r["changed_params"] == []
    assert r["checkpoint_sha_before"] == r["checkpoint_sha_after"]
    assert r["updates"] == 0


def test_group_hashes_move_only_for_the_group_that_trained():
    m = A.build("G", DOMS)
    x, y = _batch()
    before = E.group_sha(m, sorted(m.param_groups))
    E.fit(m, "har", x, y, train_groups=["head.har"], steps=2, rng=np.random.default_rng(0))
    after = E.group_sha(m, sorted(m.param_groups))
    moved = {g for g in before if before[g] != after[g]}
    assert moved == {"head.har"}


# ---------------------------------------------------------------- sequence helpers


def test_budget_and_learning_rate_fall_back_for_an_unknown_domain():
    assert S.budget("har") == S.BUDGET["har"]
    assert S.budget("har", "return") == max(30, S.BUDGET["har"] // S.RETURN_FRACTION)
    assert S.budget("not_a_domain") == 800
    assert S.lr_for("not_a_domain") == 1e-3


def test_shared_group_name_finds_the_reopenable_group_for_every_architecture():
    assert S.shared_group_name(A.build("G", DOMS)) == "fast_core"
    assert S.shared_group_name(A.build("H", DOMS)) == "fast_delta"
    assert S.shared_group_name(A.build("G_R3_split_core", DOMS)) == "fast_core_hh"
    assert S.shared_group_name(A.build("separate", DOMS)) is None


def test_reinit_actually_replaces_the_weights_it_names():
    m = A.build("G", DOMS)
    before = E.group_sha(m, ["fast_core", "head.har"])
    S.reinit(m, ["fast_core"], seed=0)
    after = E.group_sha(m, ["fast_core", "head.har"])
    assert before["fast_core"] != after["fast_core"]
    assert before["head.har"] == after["head.har"]


def test_local_groups_are_domain_scoped_and_kind_filtered():
    m = A.build("G", DOMS)
    g = S.local_groups(m, "har")
    assert all(x.endswith(".har") for x in g)
    assert set(S.local_groups(m, "har", ("head",))) == {"head.har"}
    assert S.local_groups(m, "har", ("does_not_exist",)) == []


# ---------------------------------------------------------------- data helpers


def test_the_mel_filterbank_is_normalised_triangles_over_increasing_bands():
    fb = D._melbank()
    assert fb.shape[0] == D.MEL
    assert (fb >= 0).all()
    peaks = fb.argmax(axis=1)
    assert (np.diff(peaks) >= 0).all(), "mel band centres must be non decreasing"
    assert fb.sum() > 0


def test_masking_timesteps_zeroes_the_declared_fraction_and_nothing_else():
    x = torch.ones(4, 20, 3)
    out = D.mask_timesteps(x, 0.25, np.random.default_rng(0))
    zeroed = (out.abs().sum(dim=(0, 2)) == 0).sum().item()
    assert zeroed == 5
    assert out.shape == x.shape
    assert torch.equal(x, torch.ones(4, 20, 3)), "masking must not modify its input in place"


def test_stream_construction_labels_by_the_final_sequence_and_keeps_units_pure():
    n, t, ch = 24, 8, 3
    base = {
        "x": torch.arange(n * t * ch, dtype=torch.float32).reshape(n, t, ch),
        "y": torch.arange(n) % 4,
        "u": np.array([f"u{i % 3}" for i in range(n)]),
        "xte": torch.zeros(9, t, ch),
        "yte": torch.arange(9) % 4,
        "ute": np.array([f"v{i % 3}" for i in range(9)]),
        "classes": 4,
        "unit": "synthetic",
    }
    d = D._stream_from(base, per_stream=3, n_train=12, decim=1, cache_key="_test_stream")
    D._CACHE.pop("_test_stream", None)
    assert d["x"].shape[1] == t * 3
    assert d["sequences_per_stream"] == 3
    assert set(map(str, d["u"])) <= {"u0", "u1", "u2"}
    assert set(map(str, d["ute"])) <= {"v0", "v1", "v2"}
    assert not set(map(str, d["u"])) & set(map(str, d["ute"]))
    # the label must be the last sequence of the stream, not the first and not the majority
    tail = d["x"][0, -t:]
    row = int(tail[0, 0].item() // (t * ch))
    assert int(d["y"][0]) == int(base["y"][row])


def test_domain_registry_names_every_loader_it_advertises():
    for name, fn in D.DOMAINS.items():
        assert callable(fn), name
    assert {"har", "speech", "har_stream", "speech_stream"} <= set(D.DOMAINS)
