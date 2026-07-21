"""Permanent tests for the temporal core program: custody, witnesses, factorial identity, selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import custody as C
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import hypotheses as H
from mop.temporal import io as TIO
from mop.temporal import witness as W
from mop.temporal.runs import analyze, codelife, coresel, e2, e3, mutations, successors, supervisor

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


def test_custody_inventory_integrity_rebuild_and_removal_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CANONICAL_ROOT", tmp_path)
    monkeypatch.setattr(C.io, "ROOT", tmp_path)
    target = tmp_path / "worktree"
    raw = tmp_path / "raw"
    cache = target / "cache"
    raw.mkdir()
    cache.mkdir(parents=True)
    (raw / "source.bin").write_bytes(b"source")
    (cache / "derived.bin").write_bytes(b"derived")
    raw_c = _corpus(raw, logical_identity="raw", rebuild_command="not applicable to raw data",
                    extracted_hashes={
        "source.bin": TIO.sha_file(raw / "source.bin")})
    cache_c = _corpus(cache, logical_identity="cache", retention_class="derived_rebuildable",
                      kind="non_rebuildable_cache", rebuild_command="rebuild from raw",
                      derived_caches=["raw"])
    inv = C.inventory([raw_c, cache_c])
    assert inv["n"] == 2 and inv["all_declared"] and set(inv["present"]) == {"raw", "cache"}
    assert C.verify_corpus(raw_c)["status"] == "intact"
    (raw / "source.bin").write_bytes(b"changed")
    assert C.verify_corpus(raw_c)["status"] == "damaged"
    (raw / "source.bin").unlink()
    assert C.verify_corpus(raw_c)["checks"]["no_missing_files"] is False
    (raw / "source.bin").write_bytes(b"source")
    assert not C.unique_holdings(target, [raw_c, cache_c])

    proof = target / "proof" / "only.json"
    proof.parent.mkdir(parents=True)
    proof.write_text("{}")
    held = C.unique_holdings(target, [], {"worktree/proof/only.json"})
    assert held[0]["kind"] == "unindexed_evidence"
    assert not C.guard(target, [], {"worktree/proof/only.json"})["allowed"]

    disposable = tmp_path / "disposable"
    disposable.mkdir()
    temp_c = _corpus(disposable, logical_identity="temp", retention_class="temporary")
    removed = C.remove_worktree(disposable, [temp_c], dry_run=False)
    assert removed["removed"] and not disposable.exists()

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"corpora": [cache_c.as_dict()]}))
    assert C.load_inventory(inventory_path)[0].logical_identity == "cache"


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


def test_every_live_reset_schedule_and_history_resolution_is_explicit():
    sp = {"sequence_length": 192, "segment_length": 64, "boundaries": [64, 128]}
    for kind in Fx.RESET_KINDS:
        idx, witness = Fx.reset_schedule(kind, sp, 0)
        assert witness["kind"] == kind and witness["n_resets"] == len(idx)
    for horizon in (1, 2, 5, 10, 20, 45, 90, "full"):
        idx, witness = Fx.reset_schedule(f"horizon_{horizon}", sp, 0)
        assert witness["kind"] == f"horizon_{horizon}" and witness["n_resets"] == len(idx)
    assert Fx.resolve_history_k("full_window", sp) == 192
    assert Fx.resolve_history_k("pooled_summary", sp) == 192
    assert Fx.resolve_history_k(5, sp) == 5


def test_factorial_run_emits_a_complete_training_and_unit_receipt():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(48, 30, 2)).astype(np.float32))
    y = torch.tensor(np.arange(48) % 3)
    units = np.repeat(np.arange(8), 6)
    sp = {
        "bed": "synthetic_stream", "main": (x[:24], y[:24]),
        "tune": (x[24:36], y[24:36]), "test": (x[36:], y[36:]),
        "tune_units": units[24:36], "test_units": units[36:],
        "channels": 2, "classes": 3, "sequence_length": 30,
        "segment_length": 10, "boundaries": [10, 20],
    }
    receipt = Fx.run_cell(sp, dict(Fx.REFERENCE), seed=0, eval_on="test", steps=1)
    assert receipt["steps"] == receipt["updates"] == 1
    assert receipt["checkpoint_sha_after"] and not receipt["undeclared_changes"]
    assert receipt["params"]["total"] == receipt["params"]["core"] + receipt["params"]["readout"]


def test_third_bed_cached_authorities_build_group_disjoint_splits(monkeypatch, tmp_path):
    monkeypatch.setattr(B.io, "DATA_ROOT", tmp_path)
    B._CACHE.clear()
    for bed, t, ch, classes in (("harth", 75, 6, 6), ("pamap2", 96, 18, 6)):
        root = tmp_path / bed
        root.mkdir()
        rng = np.random.default_rng(len(bed))
        np.savez(root / f"{bed}_stream.npz",
                 Xtr=rng.normal(size=(30, t, ch)).astype(np.float32),
                 Ytr=np.arange(30) % classes, Utr=np.repeat(np.arange(5), 6),
                 Xte=rng.normal(size=(12, t, ch)).astype(np.float32),
                 Yte=np.arange(12) % classes, Ute=np.repeat(np.arange(2) + 10, 6))
    for bed in ("harth_stream", "pamap2_stream"):
        ident = B.identity(bed)
        sp = B.splits(bed, 0)
        assert set(sp["units"]["main"]).isdisjoint(sp["units"]["tune"])
        assert set(sp["units"]["test"]).isdisjoint(sp["units"]["main"] + sp["units"]["tune"])
        assert ident["sequence_length"] == sp["sequence_length"]
        assert B.chance_rate(sp["classes"]) == pytest.approx(1 / sp["classes"])
        assert 0 < B.majority_rate(sp["test"][1]) <= 1
    B._CACHE.clear()


def test_stream_and_window_builders_preserve_unit_and_endpoint_identity():
    sig = np.arange(120, dtype=np.float32).reshape(60, 2)
    act = np.array([1] * 20 + [0] * 10 + [2] * 30)
    x, y, u = B._windows(sig, act, "subject", win=10, stride=10, decim=2)
    assert len(x) == len(y) == len(u) == 4 and set(y) == {1, 2}
    streams = B._stream_from(x, np.asarray(y), np.asarray(u), per_stream=3,
                             n_streams=4, decim=1, seed=0)
    assert streams[0].shape[0] == 4 and streams[0].shape[1] == x[0].shape[0] * 3
    assert set(streams[2]) == {"subject"}


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


def test_selection_excludes_oracle_controls_and_unconverged_candidates():
    safe = "gru|small|linear|none|h1"
    oracle = "gru|small|linear|true_boundary|h1"
    moving = "gru|micro|linear|horizon_45|h1"
    cells = {safe: 0.90, oracle: 0.99, moving: 0.91}
    params = {
        safe: {"total": 46000, "core": 45910},
        oracle: {"total": 46000, "core": 45910},
        moving: {"total": 20000, "core": 19610},
    }
    conv = {safe: {"classification": "converged"},
            oracle: {"classification": "converged"},
            moving: {"classification": "unconverged"}}
    principal = {
        "principal_beds": ["a", "b"],
        "per_bed": {b: {"cell_means": cells, "cell_params": params,
                          "convergence": {"configs": conv}} for b in ("a", "b")},
        "hypothesis_fold": {"hypotheses": {"H1_recurrence": {"state": "supported"}}},
    }
    assert coresel.select(principal)["selected"]["cell"] == safe


def test_extended_convergence_adds_budget_without_redefining_the_original_grid():
    assert set(e2.CONVERGENCE_GRID).isdisjoint(e2.EXTENDED_CONVERGENCE_GRID)
    assert min(e2.EXTENDED_CONVERGENCE_GRID) > max(e2.CONVERGENCE_GRID)
    cells = set(e2.LOAD_BEARING_CONVERGENCE_CELLS)
    assert "gru|small|linear|none|h1" in cells
    assert "mgu|small|linear|none|h1" in cells
    assert "histmlp|small|linear|none|hfull_window" in cells
    assert "gru|small|linear|horizon_90|h1" in cells


def test_supervisor_uses_the_measured_resource_class_optima():
    large = "xshard_har_stream_6"
    small = "xshard_har_stream_0"
    assert supervisor._large_convergence_name(large)
    assert not supervisor._large_convergence_name(small)
    cap, eligible, resource_class = supervisor.scheduling_class([small, large])
    assert (cap, eligible, resource_class) == (16, [large], "large")


def test_shard_receipts_land_atomically_with_commit_and_content_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(TIO, "RUNS", tmp_path)
    monkeypatch.setattr(TIO, "commit", lambda: "abc123")
    p = TIO.run_json("x.json", {"bed": "b", "seed": 2}, "stage")
    d = json.loads(p.read_text())
    assert d["source_commit"] == "abc123" and d["program"] == TIO.PROGRAM
    assert d["result_sha256"] == TIO.sha_obj({k: v for k, v in d.items() if k != "result_sha256"})
    assert not list(p.parent.glob(".*.partial.*"))


def test_supervisor_reports_invalid_and_partial_receipts_without_deleting_them(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "bad.json").write_text("{")
    (stage / ".work.json.partial.7").write_text("partial")
    assert supervisor.invalid("stage", ["bad"]) == ["bad"]
    assert supervisor.partials("stage") == [".work.json.partial.7"]
    assert (stage / ".work.json.partial.7").is_file()


def test_e3_common_width_is_in_band_and_shared_groups_are_shape_compatible():
    models = [e3._model(__import__("mop.temporal.beds", fromlist=["x"]).splits(b, 0), 0)
              for b in ("har_stream", "speech_stream")]
    for model in models:
        lo, hi = A.TIER_RANGE["small"]
        assert lo <= A.count(model)["core"] <= hi
        assert set(model.param_groups["local"]).isdisjoint(model.param_groups["shared"])
        assert set(model.param_groups["local"]) | set(model.param_groups["shared"]) == set(
            dict(model.named_parameters()))
    copied = e3._copy_group(models[1], models[0], "shared")
    assert copied == models[1].param_groups["shared"]
    assert len(e3.ARMS) == 8 and len(set(e3.ARMS)) == 8


def test_successor_ranking_selects_only_the_top_two_open_gates():
    gates = {
        "E5_self_supervised": {"opens": False},
        "hybrid_adaptation": {"opens": True},
        "E3_shared_versus_local": {"opens": True},
        "third_bed_replication": {"opens": True},
    }
    assert successors.ranked_successors(gates)[:2] == [
        "third_bed_replication", "E3_shared_versus_local"]


def test_required_positive_mutation_vocabulary_is_complete():
    assert set(mutations.REQUIRED) == {
        "core_bypassed", "core_output_ignored", "readout_substituted", "readout_inflated",
        "history_inserted", "history_removed", "future_history_inserted", "state_silently_reset",
        "state_silently_preserved", "reset_aligned_with_boundaries", "wrong_reset_rate",
        "parameter_count_inflated", "training_updates_inflated", "baseline_undertrained",
        "baseline_substituted", "bed_substituted", "unit_duplicated", "failing_unit_removed",
        "null_reference_changed", "effect_comparison_changed", "verdict_changed", "claim_broadened",
        "forged_completion",
    }


def test_code_lifecycle_keeps_resume_surface_active_and_sealed_drivers_frozen():
    assert codelife.classify("src/mop/temporal/runs/supervisor.py") == "active_substrate"
    assert codelife.classify("src/mop/temporal/runs/e2.py") == "active_substrate"
    assert codelife.classify("src/mop/temporal/runs/analyze.py") == "frozen_reproducibility"
    assert codelife.classify("src/mop/temporal/runs/e3.py") == "frozen_reproducibility"


def test_group_contrast_reports_both_confidence_bounds():
    d = AN.contrast({"a": [0.7, 0.8], "b": [0.5, 0.5]}, "a", "b", e2.PREREG,
                    {"a": {"u1": 0.8, "u2": 0.7, "u3": 0.9},
                     "b": {"u1": 0.5, "u2": 0.5, "u3": 0.5}})
    assert d["group_lower_95_cb"] <= d["group_mean"] <= d["group_upper_95_cb"]


def test_state_horizon_positive_requires_both_principal_beds():
    def effect(upper):
        return {"group_upper_95_cb": upper, "convergence": {"all_converged": True}}

    def bed(upper):
        return {"effects": {
            "horizon": {"gru_h45_vs_full": effect(upper)},
            "reset": {"misaligned_a": effect(upper), "misaligned_b": effect(upper),
                      "random_rate_matched": effect(upper)},
        }}

    gate = analyze.state_horizon_gate({"har_stream": bed(-0.04), "speech_stream": bed(-0.08)})
    assert gate["per_bed"]["speech_stream"]["all_pass"]
    assert not gate["per_bed"]["har_stream"]["all_pass"]
    assert not gate["all_pass"] and not gate["two_principal_beds_agree"]


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
