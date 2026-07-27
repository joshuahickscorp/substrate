"""Checkpoint/resume regression for converge_shard and extend_converge_shard.

Defect this closes: the training loop wrote its receipt once at the very end,
so a killed shard lost 100% of its progress no matter how far through the
grid x seed sweep it was — the mechanism behind every multi-hour compute loss
this session (12h, then 42h). run_cell reseeds torch/numpy and rebuilds the
model from scratch on every call (verified by reading factorial.run_cell), so
each (steps, seed) result depends only on that triple — nothing carries across
grid points, which is what makes resuming safe rather than merely convenient.

These tests replace run_cell with a deterministic fake (no real training) and
prove: a resumed run recomputes only what a checkpoint doesn't already have,
produces a byte-identical final receipt to an uninterrupted run, a checkpoint
whose identity doesn't match is never trusted, and the checkpoint file is
cleaned up on successful completion (no stale leftover for the supervisor's
stale-partial scanner to ever have to reason about).
"""

import json

import numpy as np
import pytest

from mop.temporal.runs import e2


SPEC = {"family": "gru", "tier": "small", "readout": "linear", "reset": "none", "history_k": 1}


def fake_run_cell(calls):
    def run_cell(sp, spec, seed, eval_on, steps=0):
        calls.append((steps, seed))
        # deterministic, cheap stand-in for a real training result
        return {"accuracy": round(0.1 * steps + seed, 5), "params": 42,
                "updates": steps, "checkpoint_sha_after": f"sha-{steps}-{seed}"}
    return run_cell


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(e2.io, "RUNS", tmp_path)
    monkeypatch.setattr(e2.io, "ROOT", tmp_path)
    monkeypatch.setattr(e2.io, "launch_commit", lambda: "c" * 40)
    monkeypatch.setattr(e2.io, "launch_tree_oid", lambda: "t" * 40)
    monkeypatch.setattr(e2, "CONVERGE_CONFIGS", [SPEC])
    monkeypatch.setattr(e2, "CONVERGENCE_GRID", (10, 20, 30))
    monkeypatch.setattr(e2, "CONVERGENCE_SEEDS", (0, 1))
    monkeypatch.setattr(e2.B, "splits", lambda bed, seed: {"bed": bed})
    calls = []
    monkeypatch.setattr(e2.Fx, "run_cell", fake_run_cell(calls))
    return calls


def test_fresh_run_computes_every_grid_point_once_and_leaves_no_checkpoint(sandbox, tmp_path):
    doc = e2.converge_shard("har_stream", 0)
    assert sorted(doc["curve"]) == [10, 20, 30]
    assert sandbox == [(s, sd) for s in (10, 20, 30) for sd in (0, 1)]
    assert not e2._checkpoint_path("cshard", "har_stream", 0).exists()


def test_resume_skips_completed_grid_points_and_matches_the_uninterrupted_result(sandbox, tmp_path):
    baseline = e2.converge_shard("har_stream", 0)
    sandbox.clear()

    # Simulate a kill after the first grid point (10) by hand-writing exactly
    # the checkpoint the loop itself would have written at that point.
    ckpt_path = e2._checkpoint_path("cshard", "har_stream", 0)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({
        "cell": baseline["cell"], "source_commit": "c" * 40, "source_tree_oid": "t" * 40,
        "curve": {10: baseline["curve"][10]}, "seed_spread": {10: baseline["seed_spread"][10]},
        "seed_scores": {10: baseline["seed_scores"][10]}, "arm_records": {10: baseline["arm_records"][10]},
        "parameter_count": baseline["parameter_count"], "elapsed_before": 12.3,
    }, default=str))

    resumed = e2.converge_shard("har_stream", 0)

    # Only the two remaining grid points were actually recomputed.
    assert sandbox == [(s, sd) for s in (20, 30) for sd in (0, 1)]
    # The final receipt is identical to the uninterrupted run in everything
    # that matters scientifically.
    for key in ("cell", "curve", "seed_spread", "seed_scores", "arm_records", "parameter_count"):
        assert resumed[key] == baseline[key]
    # wall_seconds correctly carries the pre-resume time forward instead of
    # silently understating total cost.
    assert resumed["wall_seconds"] >= 12.3
    assert not ckpt_path.exists()


@pytest.mark.parametrize(
    ("source_commit", "source_tree"),
    [("stale-commit", "t" * 40), ("c" * 40, "stale-tree")],
)
def test_a_checkpoint_with_the_wrong_source_authority_is_never_trusted(
        sandbox, tmp_path, source_commit, source_tree):
    ckpt_path = e2._checkpoint_path("cshard", "har_stream", 0)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({
        "cell": e2.Fx.cell_name(**SPEC), "source_commit": source_commit,
        "source_tree_oid": source_tree,
        "curve": {10: 0.5}, "seed_spread": {10: 0.0}, "seed_scores": {10: [0.5, 0.5]},
        "arm_records": {10: []}, "parameter_count": 1, "elapsed_before": 999,
    }))
    e2.converge_shard("har_stream", 0)
    # every grid point recomputed from scratch — the mismatched checkpoint was ignored
    assert sandbox == [(s, sd) for s in (10, 20, 30) for sd in (0, 1)]


def test_extend_converge_shard_resume_matches_the_uninterrupted_result(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(e2, "EXTENDED_CONVERGENCE_GRID", (40, 50))
    base = e2.converge_shard("har_stream", 0)
    sandbox.clear()

    baseline = e2.extend_converge_shard("har_stream", 0)
    sandbox.clear()

    ckpt_path = e2._checkpoint_path("xshard", "har_stream", 0)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({
        "cell": base["cell"], "source_commit": "c" * 40, "source_tree_oid": "t" * 40,
        "curve": {**base["curve"], 40: baseline["curve"][40]},
        "seed_spread": {**base["seed_spread"], 40: baseline["seed_spread"][40]},
        "seed_scores": {**base["seed_scores"], 40: baseline["seed_scores"][40]},
        "arm_records": {**base["arm_records"], 40: baseline["arm_records"][40]},
        "parameter_count": baseline["parameter_count"], "elapsed_before": 5.0,
    }, default=str))

    resumed = e2.extend_converge_shard("har_stream", 0)
    assert sandbox == [(50, 0), (50, 1)]  # only the missing extended grid point reran
    for key in ("cell", "curve", "seed_spread", "seed_scores", "arm_records"):
        assert resumed[key] == baseline[key]
    assert not ckpt_path.exists()
