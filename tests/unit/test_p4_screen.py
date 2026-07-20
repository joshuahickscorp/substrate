import json
from pathlib import Path

import torch

from mop.substrate.custom_workbench import estimated_train_step_flops
from mop.substrate.p4_screen import (
    P4_CELLS,
    P4_SERIAL_ORDER,
    P4CellSpec,
    build_substrate,
    corpus_spec_for_cell,
    estimated_train_step_flops_p4,
    model_spec_for_cell,
    realized_parameter_count,
    run_p4_screen,
    sample_random_search,
)

EXPECTED_PARAMETERS = {
    "C01": 1_646_080,
    "C02": 1_682_800,
    "C03": 1_684_808,
    "C04": 3_337_208,
    "C05": 3_291_200,
    "C06": 3_228_688,
    "C07": 6_575_616,
    "C08": 6_583_424,
    "C09": 6_636_704,
    "C10": 836_800,
    "C11": 836_880,
    "C12": 869_142,
}


def _cell(cell_id: str) -> P4CellSpec:
    return next(cell for cell in P4_CELLS if cell.cell_id == cell_id)


def test_registered_cells_and_serial_order_are_consistent():
    assert [cell.cell_id for cell in P4_CELLS] == sorted(EXPECTED_PARAMETERS)
    assert sorted(P4_SERIAL_ORDER) == sorted(EXPECTED_PARAMETERS)
    for cell in P4_CELLS:
        cell.validate()


def test_realized_parameter_counts_match_the_design_table_exactly():
    realized = {cell.cell_id: realized_parameter_count(cell) for cell in P4_CELLS}
    assert realized == EXPECTED_PARAMETERS


def test_build_substrate_forward_is_finite_with_correct_shapes():
    for cell_id in ("C01", "C11", "C12"):
        cell = _cell(cell_id)
        model = build_substrate(cell)
        clips = torch.rand(2, 3, cell.frames, 256, 256)
        with torch.no_grad():
            tokens = model.encode(clips)
            pooled = model(clips)
        expected_tokens = (cell.frames // 2) * (256 // 32) ** 2
        assert tokens.shape == (2, expected_tokens, cell.dim)
        assert pooled.shape == (2, cell.dim)
        assert torch.isfinite(tokens).all()
        assert torch.isfinite(pooled).all()


def test_dense_flops_path_equals_the_workbench_estimator_exactly_at_c01():
    cell = _cell("C01")
    data = corpus_spec_for_cell(cell)
    spec = model_spec_for_cell(cell)
    for objective in ("predictive", "random_target"):
        assert estimated_train_step_flops_p4(
            data, cell, batch_size=4, objective=objective
        ) == estimated_train_step_flops(data, spec, batch_size=4, objective=objective)


def test_predictive_and_random_target_flops_are_identical_for_every_cell():
    for cell in P4_CELLS:
        data = corpus_spec_for_cell(cell)
        predictive = estimated_train_step_flops_p4(data, cell, batch_size=4, objective="predictive")
        control = estimated_train_step_flops_p4(data, cell, batch_size=4, objective="random_target")
        assert predictive == control
        assert predictive > 0


def test_sample_random_search_is_deterministic_and_respects_the_box():
    first = sample_random_search(seed=4407, n=12)
    second = sample_random_search(seed=4407, n=12)
    assert first == second
    assert [cell.cell_id for cell in first] == [f"R{index:02d}" for index in range(1, 13)]
    for cell in first:
        cell.validate()
        assert cell.budget_label == "random"
        assert 64 <= cell.dim <= 288
        assert cell.dim % 4 == 0
        assert 2 <= cell.depth <= 8
        assert cell.frames in (8, 16)
        assert cell.corpus_variant in ("low", "high")
        if cell.family == "sparse":
            assert cell.experts in (2, 4, 8)
        else:
            assert cell.experts == 0
    assert sample_random_search(seed=1, n=12) != first


def test_run_p4_screen_two_step_smoke_writes_receipts_and_refuses_promotion(tmp_path: Path):
    cell = P4CellSpec(
        cell_id="T01",
        budget_label="1x",
        family="dense",
        dim=32,
        depth=1,
        experts=0,
        frames=8,
        corpus_variant="low",
    )
    config = {
        "profile": "unit-smoke",
        "training": {
            "seeds": [0],
            "steps": 2,
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
    run_dir = tmp_path / "p4"
    receipt = run_p4_screen(
        config,
        run_dir,
        "cpu",
        cells=[cell],
        corpus_overrides={"replicates": 3, "factor_a_levels": 4, "factor_b_levels": 4},
    )
    cell_receipt = json.loads((run_dir / "cells/T01/cell_receipt.json").read_text())
    assert cell_receipt["schema"] == "mop-p4-cell/v1"
    assert cell_receipt["complete"]
    assert not cell_receipt["futility_truncated"]
    assert cell_receipt["compute_match"]["all_ok"]
    assert cell_receipt["scores"]["predictive"]["n"] == 1
    assert (run_dir / "cells/T01/seed_0/seed_result.json").is_file()
    screen = json.loads((run_dir / "p4_screen_receipt.json").read_text())
    assert screen["schema"] == "mop-p4-screen/v1"
    assert screen["complete"] and screen["resumable"] is False
    assert screen["promotion"]["confirmatory_promotable"] is False
    assert "refused by construction" in screen["promotion"]["reason"]
    assert screen["all_ok"] is False
    assert any("response surface not estimable" in problem for problem in screen["problems"])
    assert receipt["complete"] and receipt["config_sha256"] == screen["config_sha256"]
    resumed = run_p4_screen(
        config,
        run_dir,
        "cpu",
        cells=[cell],
        corpus_overrides={"replicates": 3, "factor_a_levels": 4, "factor_b_levels": 4},
    )
    assert resumed["complete"]
    assert resumed["cells"]["T01"]["complete"]
