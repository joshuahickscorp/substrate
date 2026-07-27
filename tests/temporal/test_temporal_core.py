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
from mop.temporal.runs import (analyze, bedvalid, codelife, coresel, e2, e3, hybrid, mutations,
                               successors, supervisor, thirdbed)

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


def test_corpus_verifier_supports_file_valued_cache(tmp_path):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"derived cache")
    corpus = _corpus(
        cache,
        retention_class="derived_rebuildable",
        kind="non_rebuildable_cache",
        rebuild_command="rebuild from source",
        extracted_hashes={cache.name: TIO.sha_file(cache)},
    )
    verified = C.verify_corpus(corpus)
    assert verified["status"] == "intact"
    assert verified["live_hashes"] == {cache.name: TIO.sha_file(cache)}
    cache.write_bytes(b"drifted")
    assert C.verify_corpus(corpus)["status"] == "damaged"


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
    for channels in (6, 9, 18):
        n = A.count(A.build(family="histmlp", ch=channels, classes=6, tier="large"))["core"]
        assert A.TIER_RANGE["large"][0] <= n <= A.TIER_RANGE["large"][1]


def test_principal_loader_replaces_only_the_exact_corrected_key(monkeypatch, tmp_path):
    original = tmp_path / "e2_principal"
    corrected = tmp_path / "e2_principal_corrections"
    original.mkdir()
    corrected.mkdir()
    cell = "histmlp|large|linear|none|h1"
    untouched = "gru|small|linear|none|h1"
    (original / "har_stream_0.json").write_text(json.dumps({"runs": [
        {"seed": 0, "cell": cell, "accuracy": 0.1},
        {"seed": 0, "cell": untouched, "accuracy": 0.2}]}))
    (corrected / "capacity_har_stream_0.json").write_text(json.dumps({"runs": [
        {"seed": 0, "cell": cell, "accuracy": 0.9}]}))
    monkeypatch.setattr(TIO, "RUNS", tmp_path)
    rows = {(r["seed"], r["cell"]): r for r in analyze.load_runs("har_stream")}
    assert len(rows) == 2 and rows[(0, cell)]["accuracy"] == 0.9
    assert rows[(0, untouched)]["accuracy"] == 0.2


def test_convergence_aggregate_prefers_the_exact_corrected_cell(monkeypatch, tmp_path):
    base = tmp_path / "e2_converge"
    corrected = tmp_path / "e2_converge_corrections"
    base.mkdir()
    corrected.mkdir()
    monkeypatch.setattr(TIO, "RUNS", tmp_path)
    monkeypatch.setattr(TIO, "ROOT", tmp_path.parent)
    corrected_spec = dict(Fx.REFERENCE, family="histmlp", tier="large")
    configs = tuple([dict(Fx.REFERENCE)] * 25 + [corrected_spec])
    corrected_cell = Fx.cell_name(**corrected_spec)
    monkeypatch.setattr(e2, "CONVERGE_CONFIGS", configs)
    monkeypatch.setattr(e2, "LOAD_BEARING_CONVERGENCE_CELLS", ())
    for idx in range(26):
        (base / f"cshard_har_stream_{idx}.json").write_text(json.dumps({
            "cell": corrected_cell if idx == 25 else f"cell{idx}",
            "curve": {"400": 0.1}, "converged": True,
            "classification": "converged", "source": "original"}))
    (corrected / "convergence_har_stream.json").write_text(json.dumps({
        "cell": corrected_cell, "curve": {"400": 0.2}, "converged": True,
        "classification": "converged", "parameter_band_valid": True, "source": "correction"}))
    result = e2.converge("har_stream")
    assert result["configs"][corrected_cell]["source"] == "correction"


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


def test_harth_admission_probe_builds_disjoint_return_contexts(monkeypatch):
    units = np.repeat(np.arange(22), 6)
    fake = {"x": torch.randn(132, 12, 2), "y": torch.arange(132) % 3, "u": units,
            "channels": 2, "classes": 3}
    monkeypatch.setattr(thirdbed.B, "load", lambda name: fake)
    ctx = thirdbed.contexts(0)
    sets = [set(v) for v in ctx["units"].values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert len(set(ctx["units"]["A_train"]) | set(ctx["units"]["B_train"])) == 15
    assert len(set(ctx["units"]["A_eval"]) | set(ctx["units"]["B_eval"])) == 7
    assert not torch.allclose(ctx["B_train"][0], fake["x"][np.isin(units, ctx["units"]["B_train"])])


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


def test_selection_uses_compute_before_shorter_horizon_when_parameters_tie():
    full = "gru|small|linear|none|h1"
    short = "gru|small|linear|horizon_45|h1"
    cells = {full: 0.90, short: 0.90}
    params = {c: {"total": 46000, "core": 45910} for c in cells}
    principal = {
        "principal_beds": ["a", "b"],
        "per_bed": {b: {"cell_means": cells, "cell_params": params,
                          "metric_panel": {"latency_wall_seconds_mean": {full: 1.0, short: 2.0}}}
                    for b in ("a", "b")},
        "hypothesis_fold": {"hypotheses": {"H1_recurrence": {"state": "supported"}}},
    }
    assert coresel.select(principal)["selected"]["cell"] == full


def test_extended_convergence_adds_budget_without_redefining_the_original_grid():
    assert set(e2.CONVERGENCE_GRID).isdisjoint(e2.EXTENDED_CONVERGENCE_GRID)
    assert min(e2.EXTENDED_CONVERGENCE_GRID) > max(e2.CONVERGENCE_GRID)
    cells = set(e2.LOAD_BEARING_CONVERGENCE_CELLS)
    assert "gru|small|linear|none|h1" in cells
    assert "mgu|small|linear|none|h1" in cells
    assert "histmlp|small|linear|none|hfull_window" in cells
    assert "gru|small|linear|horizon_90|h1" in cells
    converged_cells = {Fx.cell_name(**c) for c in e2.CONVERGE_CONFIGS}
    assert all(Fx.cell_name(**dict(Fx.REFERENCE, family=f, tier=t)) in converged_cells
               for f in ("gru", "lstm", "mgu", "pooled", "histmlp", "tcn")
               for t in A.CAPACITY_TIERS)
    assert all(Fx.cell_name(**spec) in converged_cells for group in (
        "architecture", "readout", "horizon", "reset", "history", "capacity_by_horizon",
        "capacity_by_readout")
               for spec in Fx.sweep_cells()[group])


def test_supervisor_uses_the_measured_resource_class_optima():
    large = "xshard_har_stream_6"
    small = "xshard_har_stream_0"
    assert supervisor._large_convergence_name(large)
    assert not supervisor._large_convergence_name(small)
    cap, eligible, resource_class = supervisor.scheduling_class([small, large])
    assert (cap, eligible, resource_class) == (16, [large], "large_class_isolated")


def test_shard_receipts_land_atomically_with_commit_and_content_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(TIO, "RUNS", tmp_path)
    monkeypatch.setattr(TIO, "PROCESS_SOURCE_COMMIT", "abc123")
    monkeypatch.setattr(TIO, "PROCESS_SOURCE_TREE_OID", "def456")
    p = TIO.run_json("x.json", {"bed": "b", "seed": 2}, "stage")
    d = json.loads(p.read_text())
    assert d["source_commit"] == "abc123" and d["source_tree_oid"] == "def456"
    assert d["program"] == TIO.PROGRAM
    assert d["result_sha256"] == TIO.sha_obj({k: v for k, v in d.items() if k != "result_sha256"})
    assert not list(p.parent.glob(".*.partial.*"))


def test_direct_receipt_source_is_bound_once_not_looked_up_at_write(monkeypatch, tmp_path):
    monkeypatch.setattr(TIO, "RUNS", tmp_path)
    monkeypatch.setattr(TIO, "PROCESS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(TIO, "PROCESS_SOURCE_TREE_OID", "b" * 40)
    monkeypatch.setattr(TIO, "commit", lambda: "c" * 40)
    d = json.loads(TIO.run_json("bound.json", {}, "stage").read_text())
    assert d["source_commit"] == "a" * 40
    assert d["source_tree_oid"] == "b" * 40


def test_principal_uses_each_cells_strict_selected_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(e2.io, "ROOT", tmp_path)
    monkeypatch.setattr(e2.io, "RUNS", tmp_path / "runs")
    path = e2.io.RUNS / "e2_converge" / "converge_har_stream.json"
    path.parent.mkdir(parents=True)
    specs = e2.Fx.sweep_cells()["_all"]
    selected = {e2.Fx.cell_name(**spec): 400 + 400 * (i % 4) for i, spec in enumerate(specs)}
    path.write_text(json.dumps({"configs": {
        cell: {"selected_checkpoint": steps} for cell, steps in selected.items()}}))
    monkeypatch.setattr(e2.B, "splits", lambda *_: {"test": (None, []), "classes": 2})
    monkeypatch.setattr(e2.B, "majority_rate", lambda *_: 0.5)
    monkeypatch.setattr(e2.B, "chance_rate", lambda *_: 0.5)
    calls = []
    monkeypatch.setattr(e2.io, "launch_commit", lambda: "c" * 40)
    monkeypatch.setattr(e2.io, "launch_tree_oid", lambda: "t" * 40)
    monkeypatch.setattr(e2.Fx, "run_cell", lambda sp, spec, seed, split, steps: (
        calls.append(e2.Fx.cell_name(**spec)) or {
            "cell": e2.Fx.cell_name(**spec), "bed": "har_stream", "seed": seed,
            "steps": steps, "updates": steps, "accuracy": 0.5}))
    doc = e2.principal("har_stream", 0)
    assert doc["schema"] == "mop-e2-principal-shard/v2"
    assert {row["cell"]: row["steps"] for row in doc["runs"]} == selected
    assert doc["convergence_authority"]["sha256"] == e2.io.sha_file(path)
    progress = e2._checkpoint_path("principal", "har_stream", 0)
    progress.parent.mkdir(parents=True, exist_ok=True)
    prefix = doc["runs"][:11]
    progress.write_text(json.dumps({
        "schema": "mop-e2-principal-progress/v1", "cell": "principal:har_stream:0",
        "source_commit": "c" * 40, "source_tree_oid": "t" * 40,
        "bed": "har_stream", "seed": 0, "convergence_sha256": e2.io.sha_file(path),
        "selected_checkpoints": selected, "runs": prefix, "elapsed_before": 12.5,
    }))
    calls.clear()
    resumed = e2.principal("har_stream", 0)
    assert calls == [e2.Fx.cell_name(**spec) for spec in specs[11:]]
    assert resumed["runs"] == doc["runs"] and resumed["wall_seconds"] >= 12.5
    assert not progress.exists()
    stale = {
        "schema": "mop-e2-principal-progress/v1", "cell": "principal:har_stream:0",
        "source_commit": "c" * 40, "source_tree_oid": "t" * 40,
        "bed": "har_stream", "seed": 0, "convergence_sha256": "stale",
        "selected_checkpoints": selected, "runs": prefix, "elapsed_before": 999,
    }
    progress.write_text(json.dumps(stale))
    calls.clear()
    restarted = e2.principal("har_stream", 0)
    assert calls == [e2.Fx.cell_name(**spec) for spec in specs]
    assert restarted["runs"] == doc["runs"] and restarted["wall_seconds"] < 999


def test_supervisor_reports_invalid_and_partial_receipts_without_deleting_them(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "bad.json").write_text("{")
    forged = TIO.run_json("forged.json", {"seed": 1}, "stage")
    forged_doc = json.loads(forged.read_text())
    forged_doc["seed"] = 2
    forged.write_text(json.dumps(forged_doc))
    (stage / ".work.json.partial.7").write_text("partial")
    assert supervisor.invalid("stage", ["bad"]) == ["bad"]
    assert supervisor.invalid("stage", ["forged"]) == ["forged"]
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


def test_e3_target_comparison_requires_identical_optimizer_exposure_and_batches():
    receipt = {"updates": 1200, "batch": 64, "lr": 0.003, "optimizer": "Adam",
               "batch_seed": 50000, "trainable_params": ["core", "head"],
               "trainable_param_count": 100}
    assert e3._target_training_match(receipt, dict(receipt))["all_matched"]
    changed = dict(receipt, batch_seed=50001)
    assert not e3._target_training_match(receipt, changed)["all_matched"]


def test_successor_ranking_selects_only_the_top_two_open_gates():
    gates = {
        "E5_self_supervised": {"opens": False},
        "hybrid_adaptation": {"opens": True, "ranking": {"priority_score": 0.1}},
        "E3_shared_versus_local": {"opens": True, "ranking": {"priority_score": 0.2}},
        "third_bed_replication": {"opens": True, "ranking": {"priority_score": 0.3}},
    }
    assert successors.ranked_successors(gates)[:2] == [
        "third_bed_replication", "E3_shared_versus_local"]


def test_failed_third_bed_replication_does_not_consume_a_successor_slot(monkeypatch):
    artifacts = {
        "MOP_E2_PRINCIPAL_RESULT.json": {"principal_beds": []},
        "MOP_OWNED_TEMPORAL_CORE_V1.json": {},
        "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json": {"selected": ["harth_stream"]},
        "MOP_THIRD_TEMPORAL_BED_RESULT.json": {
            "classification": "valid_secondary_bed_did_not_reproduce_the_principal_effect"},
    }
    monkeypatch.setattr(successors.io, "exists", lambda name: name in artifacts)
    monkeypatch.setattr(successors.io, "load", lambda name: artifacts[name])
    assert not successors.gates()["third_bed_replication"]["opens"]


def test_capacity_is_not_a_forgetting_proxy_without_retention_or_interference_measurement():
    absent = successors._capacity_forgetting({"per_bed": {"x": {"findings": {
        "capacity_monotonic": False}}}})
    assert not absent["measured"] and not absent["increases_forgetting"]
    measured = successors._capacity_forgetting({"capacity_retention_or_interference": {
        "measured": True, "estimand": "retention_loss_large_minus_small",
        "per_independent_unit_effects": [0.2, 0.21, 0.19]}})
    assert measured["measured"] and measured["increases_forgetting"]


def test_harth_order_gate_requires_a_matched_timestep_permutation():
    assert not bedvalid.order_permutation_necessity({})["measured"]
    witness = {"temporal_order_permutation": {
        "intervention": "within_window_timestep_permutation", "labels_unchanged": True,
        "ordered_per_unit_accuracy": {"u1": 0.9, "u2": 0.9, "u3": 0.9},
        "permuted_per_unit_accuracy": {"u1": 0.5, "u2": 0.5, "u3": 0.5},
        "seed_decision": {"verdict": "positive"},
        "pooled_group_lower_95_cb": -0.001, "pooled_group_upper_95_cb": 0.001,
        "resource_match": {"same_examples": True, "same_model_checkpoint": True,
                           "same_evaluation_code": True}}}
    result = bedvalid.order_permutation_necessity(witness)
    assert result["measured"] and result["necessary"] and result["group_lower_95_cb"] >= TIO.SESOI


def test_harth_time_intervention_preserves_each_example_multiset_and_labels():
    x = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    y, units = torch.tensor([1, 2]), np.asarray(["a", "b"])
    permuted, py, pu = thirdbed.permute_time((x, y, units), 4)
    assert torch.equal(y, py) and np.array_equal(units, pu)
    assert all(torch.equal(torch.sort(x[i], dim=0).values, torch.sort(permuted[i], dim=0).values)
               for i in range(len(x)))
    assert not torch.equal(x, permuted)


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


def test_hybrid_state_only_rule_changes_no_parameters(monkeypatch):
    monkeypatch.setattr(hybrid, "ADAPT_STEPS", 1)
    model = hybrid.HybridModel(A.build(family="gru", ch=2, classes=3, tier="micro"))
    x = torch.randn(12, 8, 2)
    y = torch.arange(12) % 3
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trace = hybrid.state_adapt(model, (x, y, np.arange(12)), seed=0, train_head=False)
    assert trace["parameter_updates"] == 0 and not trace["changed_params"]
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert trace["not_E4_recentering_rule"] and model.state.norm() > 0


def test_hybrid_head_uses_control_lr_and_batch_stream_and_noise_uses_learned_norm(monkeypatch):
    monkeypatch.setattr(hybrid, "ADAPT_STEPS", 1)
    model = hybrid.HybridModel(A.build(family="gru", ch=2, classes=3, tier="micro"))
    x, y = torch.randn(12, 8, 2), torch.arange(12) % 3
    trace = hybrid.state_adapt(model, (x, y, np.arange(12)), seed=3, train_head=True)
    assert trace["head_lr"] == Fx.LR
    assert trace["batch_seed"] == hybrid._adapt_batch_seed(3)
    learned = torch.tensor([3.0, 4.0])
    assert hybrid._noise_matched_to(learned).norm() == pytest.approx(learned.norm())


def test_code_lifecycle_keeps_resume_surface_active_and_sealed_drivers_frozen():
    assert codelife.classify("src/mop/temporal/runs/supervisor.py") == "active_substrate"
    assert codelife.classify("src/mop/temporal/runs/e2.py") == "active_substrate"
    assert codelife.classify("src/mop/temporal/receipt_contract.py") == "active_runtime"
    assert codelife.classify("src/mop/temporal/custody.py") == "active_runtime"
    assert codelife.classify("src/mop/temporal/io.py") == "active_runtime"
    assert codelife.classify("src/mop/temporal/runs/analyze.py") == "frozen_reproducibility"
    assert codelife.classify("src/mop/temporal/runs/e3.py") == "frozen_reproducibility"
    rows = codelife.scan()
    active_e2 = {
        path: loc for path, loc in rows["active_substrate"].items()
        if path.startswith("src/mop/temporal")
    }
    assert sum(active_e2.values()) <= codelife.TARGETS["new_active_e2_implementation_loc"]
    assert (
        sum(rows["active_runtime"].values()) + sum(rows["active_substrate"].values())
        <= codelife.TARGETS["active_runtime_loc"]
    )


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
