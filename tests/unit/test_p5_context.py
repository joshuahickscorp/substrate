import json
from pathlib import Path

import pytest
import torch
import yaml

from mop.config import REPO_ROOT
from mop.substrate.custom_workbench import (
    WorkbenchRefused,
    estimated_train_step_flops,
    parameter_count,
)
from mop.substrate.p5_context import (
    P5_CELLS,
    P5_FRAME_COUNTS,
    P5_MECHANISMS,
    P5CellSpec,
    WindowedBlocks,
    _unwindows,
    _verify_config_cells,
    _windows,
    build_p5_substrate,
    corpus_spec_for_frames,
    estimated_train_step_flops_p5,
    model_spec_for_cell,
    run_p5_pilot,
    solve_matched_steps,
)

# Known-answer design table: realized trainable parameter counts per frame count. The three
# transformer mechanisms are parameter identical; the depth-8 GRU stack carries the designed
# 512-parameter deficit on the blocks (793088 transformer block parameters versus 792576).
EXPECTED_TRANSFORMER_PARAMETERS = {16: 1_678_848, 32: 1_744_384, 64: 1_875_456}
EXPECTED_RECURRENT_DEFICIT = 512

# Known-answer FLOP table at f64, batch 4 (the pilot's training batch size).
EXPECTED_F64_FLOPS_PER_STEP = {
    "exact_global": 242_131_009_536,
    "window_local": 139_051_794_432,
    "recurrent": 104_692_187_136,
    "hierarchical_pooled": 139_220_090_880,
}


def test_registered_cells_are_the_twelve_frame_mechanism_pairs():
    assert len(P5_CELLS) == 12
    assert len(set(P5_CELLS)) == 12
    assert {cell.frames for cell in P5_CELLS} == set(P5_FRAME_COUNTS)
    assert {cell.mechanism for cell in P5_CELLS} == set(P5_MECHANISMS)
    for cell in P5_CELLS:
        cell.validate()
    with pytest.raises(ValueError):
        P5CellSpec(8, "exact_global").validate()
    with pytest.raises(ValueError):
        P5CellSpec(16, "sparse").validate()


def test_config_cell_table_matches_the_registered_constant():
    config = yaml.safe_load((REPO_ROOT / "configs/experiment/mop_p5_context_capability.yaml").read_text())
    _verify_config_cells(config["cells"])
    drifted = [dict(row) for row in config["cells"]]
    drifted[0]["frames"] = 32
    with pytest.raises(WorkbenchRefused):
        _verify_config_cells(drifted)


def test_parameter_identity_matches_the_design_table_exactly():
    for frames in P5_FRAME_COUNTS:
        counts = {
            mechanism: parameter_count(build_p5_substrate(P5CellSpec(frames, mechanism)))
            for mechanism in P5_MECHANISMS
        }
        for mechanism in ("exact_global", "window_local", "hierarchical_pooled"):
            assert counts[mechanism] == EXPECTED_TRANSFORMER_PARAMETERS[frames]
        assert counts["exact_global"] - counts["recurrent"] == EXPECTED_RECURRENT_DEFICIT
        deviation = EXPECTED_RECURRENT_DEFICIT / counts["exact_global"]
        assert deviation < 0.005


def test_window_reshape_roundtrip_preserves_token_order():
    hidden = torch.arange(1 * 1024 * 8, dtype=torch.float32).reshape(1, 1024, 8)
    windows = _windows(hidden, 512)
    assert windows.shape == (2, 512, 8)
    assert torch.equal(windows[0], hidden[0, :512])
    assert torch.equal(windows[1], hidden[0, 512:])
    assert torch.equal(_unwindows(windows, 1), hidden)
    # A sequence at or below one window stays a single degenerate-global window.
    short = torch.randn(2, 256, 8)
    assert torch.equal(_unwindows(_windows(short, 512), 2), short)
    assert _windows(short, 512).shape == (2, 256, 8)
    # Ragged token counts are refused rather than silently padded.
    with pytest.raises(ValueError):
        _windows(torch.randn(1, 768, 8), 512)


def test_windowed_blocks_zero_depth_composes_the_reshape_as_identity():
    blocks = WindowedBlocks(8, 0, 4, 4)
    hidden = torch.randn(2, 1024, 8)
    assert torch.equal(blocks(hidden), hidden)


def test_exact_global_flops_path_equals_the_workbench_estimator():
    for frames in P5_FRAME_COUNTS:
        cell = P5CellSpec(frames, "exact_global")
        data = corpus_spec_for_frames(frames)
        spec = model_spec_for_cell(cell)
        for objective in ("predictive", "random_target"):
            assert estimated_train_step_flops_p5(
                data, cell, batch_size=4, objective=objective
            ) == estimated_train_step_flops(data, spec, batch_size=4, objective=objective)


def test_window_flops_are_degenerate_dense_at_f16():
    data = corpus_spec_for_frames(16)
    dense = estimated_train_step_flops_p5(
        data, P5CellSpec(16, "exact_global"), batch_size=4, objective="predictive"
    )
    window = estimated_train_step_flops_p5(
        data, P5CellSpec(16, "window_local"), batch_size=4, objective="predictive"
    )
    assert window == dense


def test_f64_flop_table_matches_the_known_answers_exactly():
    data = corpus_spec_for_frames(64)
    for mechanism, expected in EXPECTED_F64_FLOPS_PER_STEP.items():
        assert (
            estimated_train_step_flops_p5(
                data, P5CellSpec(64, mechanism), batch_size=4, objective="predictive"
            )
            == expected
        )


def test_solve_matched_steps_grain5_matches_every_mechanism_at_f64():
    # Preregistration amendment, made before any pilot seed ran: matching rounds to multiples of
    # step_granularity 5 (checkpoints stay at 25). Checkpoint-multiple rounding stranded the
    # recurrent arm at 2.69 percent deviation, a grid artifact rather than a compute mismatch.
    dense = EXPECTED_F64_FLOPS_PER_STEP["exact_global"]
    solved = {
        mechanism: solve_matched_steps(200, dense, EXPECTED_F64_FLOPS_PER_STEP[mechanism], 25)
        for mechanism in ("window_local", "recurrent", "hierarchical_pooled")
    }
    for row in solved.values():
        assert row["steps"] % 5 == 0
        assert row["steps"] >= 5
        assert row["step_granularity"] == 5
        assert row["checkpoint_every"] == 25
        assert row["matched_ok"]
        assert row["fractional_deviation"] <= 0.02
    assert solved["window_local"]["steps"] == 350
    assert solved["hierarchical_pooled"]["steps"] == 350
    assert solved["recurrent"]["steps"] == 465
    # Exact FLOP parity solves to the dense reference step count with zero deviation.
    identity = solve_matched_steps(200, dense, dense, 25)
    assert identity["steps"] == 200 and identity["fractional_deviation"] == 0.0
    # The solver never returns fewer than one granularity interval.
    tiny = solve_matched_steps(1, 1, 10**9, 25)
    assert tiny["steps"] == 5 and not tiny["matched_ok"]


def test_run_p5_pilot_two_step_smoke_writes_receipts_resumes_and_refuses_promotion(tmp_path: Path):
    cells = [P5CellSpec(16, "exact_global"), P5CellSpec(16, "window_local")]
    config = {
        "profile": "unit-smoke",
        "training": {
            "seeds": [0],
            "dense_steps": 2,
            "batch_size": 2,
            "eval_batch_size": 8,
            "learning_rate": 0.0005,
            "weight_decay": 0.02,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "variance_weight": 0.1,
            "checkpoint_every": 1,
            "wall_budget_seconds": 600.0,
        },
        "screen": {"sesoi": 0.10, "futility_margin": 0.10, "min_free_disk_gb": 0.0},
    }
    run_dir = tmp_path / "p5"
    receipt = run_p5_pilot(
        config,
        run_dir,
        "cpu",
        cells=cells,
        corpus_overrides={"replicates": 3},
        model_overrides={"dim": 32},
    )
    cell_receipt = json.loads((run_dir / "frames/f16/cell_receipt.json").read_text())
    assert cell_receipt["schema"] == "mop-p5-context-cell/v1"
    assert cell_receipt["complete"]
    assert not cell_receipt["futility_truncated"]
    assert cell_receipt["seeds_completed"] == 1
    for mechanism in ("exact_global", "window_local"):
        assert cell_receipt["compute"]["per_mechanism"][mechanism]["matched"]["matched_ok"]
        assert cell_receipt["scores"][mechanism]["n"] == 1
    # Windowing is degenerate-global at f16, so the matched step count equals the dense reference.
    assert cell_receipt["compute"]["per_mechanism"]["window_local"]["matched"]["steps"] == 2
    assert cell_receipt["parity_diagnostic"] is not None
    assert cell_receipt["parity_diagnostic"]["hard_gate"] is False
    assert "degenerate-global" in cell_receipt["parity_diagnostic"]["note"]
    seed_result = json.loads((run_dir / "frames/f16/seed_0/seed_result.json").read_text())
    assert seed_result["schema"] == "mop-p5-context-seed/v1"
    assert seed_result["complete"]
    accounting = json.loads((run_dir / "accounting/0.json").read_text())
    assert accounting["schema"] == "mop-p9-workload-accounting/v1"
    assert {phase["name"] for phase in accounting["phases"]} <= {
        "input",
        "model",
        "evaluation",
        "checkpoint",
    }
    top = json.loads((run_dir / "p5_context_receipt.json").read_text())
    assert top["schema"] == "mop-p5-context-screen/v1"
    assert top["complete"] and top["resumable"] is False
    assert top["trainability_gate_failed"] is False
    assert top["promotion"]["confirmatory_promotable"] is False
    assert top["promotion"]["refused_by_construction"] is True
    assert "category 9 is impossible" in top["promotion"]["category_9_statement"]
    # No f64 or f32 cells were injected, so the tier contrasts honestly fail closed.
    assert top["all_ok"] is False
    assert any("f64 primary contrasts" in problem for problem in top["problems"])
    assert receipt["complete"] and receipt["config_sha256"] == top["config_sha256"]
    # The same command resumes from durable receipts without retraining.
    resumed = run_p5_pilot(
        config,
        run_dir,
        "cpu",
        cells=cells,
        corpus_overrides={"replicates": 3},
        model_overrides={"dim": 32},
    )
    assert resumed["complete"]
    assert resumed["frames"]["f16"]["complete"]
    resumed_cell = json.loads((run_dir / "frames/f16/cell_receipt.json").read_text())
    assert resumed_cell["seed_results"]["0"].get("resumed_from_complete_receipt") is True
