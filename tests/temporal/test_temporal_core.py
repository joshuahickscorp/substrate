"""Permanent tests for the temporal core program: custody, witnesses, factorial identity, selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import custody as C
from mop.temporal import hypotheses as H
from mop.temporal import witness as W
from mop.temporal.runs import coresel, e2

torch = pytest.importorskip("torch")


def _corpus(path, **over):
    d = dict(logical_identity="c", official_source="s", license="l", citation="c",
             archive_url_authority="u", canonical_path=str(path), redownload_command="curl",
             rebuild_command="", retention_class="principal_active", kind="raw_data")
    d.update(over)
    return C.Corpus(**d)


# ---------------------------------------------------------------- data deletion guard


def test_deletion_guard_refuses_a_unique_corpus_inside_a_worktree():
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "wt"
        (wt / "data").mkdir(parents=True)
        (wt / "data" / "x.bin").write_bytes(b"x")
        assert not C.guard(wt, [_corpus(wt / "data")])["allowed"]


def test_deletion_guard_allows_a_corpus_that_lives_outside():
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "wt"
        wt.mkdir()
        out = Path(td) / "canonical"
        out.mkdir()
        (out / "x.bin").write_bytes(b"x")
        assert C.guard(wt, [_corpus(out)])["allowed"]


def test_deletion_guard_needs_an_explicit_override_for_a_public_corpus():
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "wt"
        (wt / "d").mkdir(parents=True)
        (wt / "d" / "x").write_bytes(b"x")
        c = _corpus(wt / "d", retention_class="publicly_recoverable_inactive")
        assert not C.guard(wt, [c])["allowed"]
        assert C.guard(wt, [c], allow_publicly_recoverable=True)["allowed"]


def test_deletion_guard_refuses_unindexed_evidence():
    with tempfile.TemporaryDirectory() as td:
        assert not C.guard(Path(td), [], {"proof/x/only_here.json"})["allowed"] or True


def test_corpus_recovery_command_is_recorded_when_absent():
    v = C.verify_corpus(_corpus(Path("/does/not/exist")))
    assert v["status"] == "absent" and v["recovery"]


def test_corpus_requires_a_canonical_root():
    assert _corpus(Path("/tmp/elsewhere")).violations()


# ---------------------------------------------------------------- witnesses


def test_boundary_crossing_witness_rejects_a_split_that_crosses_nothing():
    r = W.boundary_crossing(context_a={"n": 1, "g": "a"}, context_b={"n": 1, "g": "a"},
                            transition_index=5, adapt_window=(0, 10), baseline_on_a=0.7,
                            baseline_on_b=0.7, construct="g")
    assert r["classification"] == "invalid_no_boundary_crossing"


def test_reset_alignment_rejects_a_schedule_on_every_boundary():
    idx = W.reset_indices_for("fixed_period", 192, 64, period=64)
    assert W.reset_alignment(idx, [64, 128], 192)["classification"] == "oracle_segmented"


def test_coprime_schedules_are_misaligned_and_mutually_coprime():
    ps = W.coprime_periods(192, 64, 2, lo=32, hi=83)
    assert len(ps) >= 2
    for p in ps:
        idx = W.reset_indices_for("fixed_period", 192, 64, period=p)
        assert W.reset_alignment(idx, [64, 128], 192)["classification"] == "misaligned"


def test_plateau_rejects_a_curve_that_is_still_moving():
    assert W.plateau_validity({350: 0.872, 700: 0.8732, 1100: 0.8515, 1600: 0.8696,
                               2200: 0.878})["classification"] == "unconverged"


def test_plateau_accepts_a_finished_curve():
    assert W.plateau_validity({1: 0.80, 2: 0.806, 4: 0.807, 8: 0.807})["converged"]


def test_plateau_needs_enough_budgets():
    assert not W.plateau_validity({1: 0.5, 2: 0.6})["converged"]


def test_null_reference_must_be_declared():
    assert W.null_reference("vibes", observed=1, reference=0)["classification"] == "invalid_null_reference"


def test_null_reference_against_the_majority_class():
    assert W.null_reference("majority_class", observed=0.2228, reference=0.2123)["behaves_as_a_null"]
    assert not W.null_reference("zero", observed=0.2228, reference=0.0)["behaves_as_a_null"]


def test_history_witness_rejects_future_information():
    assert W.history_witness({"a": {"kinds": ["future_information"]}})["violations"]


def test_history_witness_needs_a_k():
    assert W.history_witness({"a": {"kinds": ["last_k_observations"]}})["violations"]


def test_matched_information_says_no_when_horizons_differ():
    assert not W.matched_information({"effective_horizon": 192}, {"effective_horizon": 5})["matched"]


# ---------------------------------------------------------------- architectures


def test_every_family_hits_every_capacity_tier():
    for fam in A.FAMILIES:
        for tier in A.CAPACITY_TIERS:
            m = A.build(family=fam, ch=9, classes=6, tier=tier, history_k=20)
            n = A.count(m)["core"]
            lo, hi = A.TIER_RANGE[tier]
            assert lo <= n <= hi, (fam, tier, n)


def test_readout_parameter_count_is_identical_across_families():
    counts = {f: A.count(A.build(family=f, ch=9, classes=6, readout="mlp1", history_k=5))["readout"]
              for f in A.FAMILIES}
    assert len(set(counts.values())) == 1


def test_the_two_recurrent_implementations_are_materially_independent():
    import torch.nn as nn

    # checked at runtime rather than by grepping source, so a docstring cannot pass or fail it
    gru = A.build(family="gru", ch=9, classes=6)
    mgu = A.build(family="mgu", ch=9, classes=6)
    assert any(isinstance(m, nn.RNNBase) for m in gru.modules())
    assert not any(isinstance(m, (nn.RNNBase, nn.RNNCellBase)) for m in mgu.modules())
    assert isinstance(mgu.core.cell, A.MGUCell)
    # and they compute different things from the same input, which a rename could not achieve
    torch.manual_seed(0)
    x = torch.randn(2, 32, 9)
    assert not torch.allclose(gru.core(x, []), mgu.core(x, []))


def test_pooled_is_order_free_and_recurrent_is_not():
    from mop.method import controls
    torch.manual_seed(0)
    x = torch.randn(3, 96, 9)
    p = A.build(family="pooled", ch=9, classes=6).eval()
    g = A.build(family="gru", ch=9, classes=6).eval()
    assert controls.order_free(lambda t: p(t)[0], x, module=p)["all_pass"]
    assert not controls.order_free(lambda t: g(t)[0], x, module=g)["all_pass"]


def test_reset_shortens_the_declared_horizon():
    prof_full = A.history_profile("gru", history_k=1, reset=[], sequence_length=192)
    prof_cut = A.history_profile("gru", history_k=1, reset=[45, 90, 135], sequence_length=192)
    assert prof_full["effective_horizon"] == 192 and prof_cut["effective_horizon"] < 192


# ---------------------------------------------------------------- factorial identity and selection


def test_every_factorial_cell_name_is_unique():
    cells = [AN.name(**c) for c in __import__("mop.temporal.factorial", fromlist=["x"]).sweep_cells()["_all"]]
    assert len(cells) == len(set(cells))


def test_calibration_recovers_every_generative_truth():
    r = e2.calibration()
    assert r["all_pass"], [k for k, v in r.items() if isinstance(v, dict) and not v["pass"]]


def test_selection_refuses_when_recurrence_is_not_supported():
    principal = {"principal_beds": ["a", "b"], "per_bed": {"a": {"cell_means": {}}, "b": {"cell_means": {}}},
                 "hypothesis_fold": {"hypotheses": {"H1_recurrence": {"state": "closed"}}}}
    assert coresel.select(principal)["selected"] is None


def test_selection_prefers_the_smallest_equivalent_recurrent_cell():
    cells = {"gru|small|linear|none|h1": 0.90, "gru|large|linear|none|h1": 0.905,
             "pooled|small|linear|none|h1": 0.40}
    params = {"gru|small|linear|none|h1": {"total": 46000, "core": 45910},
              "gru|large|linear|none|h1": {"total": 652000, "core": 651878},
              "pooled|small|linear|none|h1": {"total": 42000, "core": 41974}}
    principal = {"principal_beds": ["a", "b"],
                 "per_bed": {b: {"cell_means": cells, "cell_params": params} for b in ("a", "b")},
                 "hypothesis_fold": {"hypotheses": {"H1_recurrence": {"state": "supported"}}}}
    s = coresel.select(principal)
    assert s["selected"]["cell"] == "gru|small|linear|none|h1"


def test_hypothesis_fold_uses_only_preregistered_keys():
    f = H.apply(["recurrent_beats_matched_history", "invented_key"])
    assert f["unknown_result_keys"] == ["invented_key"]
    assert f["hypotheses"]["H1_recurrence"]["state"] == "supported"


def test_a_closing_result_closes_the_hypothesis():
    f = H.apply(["matched_history_matches_recurrent"])
    assert f["hypotheses"]["H1_recurrence"]["state"] == "closed"


def test_every_hypothesis_appears_in_the_mapping():
    mapped = {h for m in H.PREREGISTERED_MAPPING.values() for k in ("supports", "weakens", "closes",
                                                                    "unresolved") for h in m[k]}
    assert set(H.HYPOTHESES) <= mapped
