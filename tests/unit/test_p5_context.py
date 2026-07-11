import copy
import json
from pathlib import Path

import pytest
import scripts.p5_context_capability as p5_cli
import torch
import yaml

import mop.substrate.p5_context as p5_context
from mop.config import REPO_ROOT
from mop.substrate.custom_workbench import (
    WorkbenchRefused,
    estimated_train_step_flops,
    json_sha256,
    parameter_count,
    sha256_file,
)
from mop.substrate.p5_context import (
    P5_CELLS,
    P5_FRAME_COUNTS,
    P5_MECHANISMS,
    P5CellSpec,
    WindowedBlocks,
    _fresh_challenge_required,
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
P5_CORE_RUNTIME_SOURCES = (
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)

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


def test_run_p5_pilot_two_step_smoke_writes_receipts_resumes_and_refuses_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert top["source_bindings_sha256"] == json_sha256(top["source_bindings"])
    assert top["checkpoint_requirements_sha256"] == p5_context._checkpoint_requirements_sha256(
        top["cell_registry_sha256"], top["source_bindings_sha256"]
    )
    assert seed_result["source_bindings_sha256"] == top["source_bindings_sha256"]
    assert seed_result["checkpoint_requirements_sha256"] == top["checkpoint_requirements_sha256"]
    for mechanism in ("exact_global", "window_local"):
        arm_dir = run_dir / "frames/f16/seed_0" / mechanism
        arm_receipt = json.loads((arm_dir / "arm_receipt.json").read_text())
        checkpoint = torch.load(arm_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert arm_receipt["requirements_sha256"] == top["checkpoint_requirements_sha256"]
        assert checkpoint["requirements_sha256"] == top["checkpoint_requirements_sha256"]
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

    for relative in P5_CORE_RUNTIME_SOURCES:
        mutated_source = copy.deepcopy(top["source_bindings"])
        source_index = p5_context.P5_SOURCE_PATHS.index(relative)
        mutated_source[source_index]["file_sha256"] = "0" * 64
        with monkeypatch.context() as source_drift:
            source_drift.setattr(
                p5_context,
                "_source_bindings",
                lambda bindings=mutated_source: bindings,
            )
            with pytest.raises(WorkbenchRefused, match="source_bindings_sha256"):
                run_p5_pilot(
                    config,
                    run_dir,
                    "cpu",
                    cells=cells,
                    corpus_overrides={"replicates": 3},
                    model_overrides={"dim": 32},
                )

    mutated_bindings = copy.deepcopy(top["source_bindings"])
    mutated_bindings[-1]["file_sha256"] = "0" * 64
    monkeypatch.setattr(p5_context, "_source_bindings", lambda: mutated_bindings)
    with pytest.raises(WorkbenchRefused, match="source_bindings_sha256"):
        run_p5_pilot(
            config,
            run_dir,
            "cpu",
            cells=cells,
            corpus_overrides={"replicates": 3},
            model_overrides={"dim": 32},
        )

    (run_dir / "frames/f16/seed_0/seed_result.json").unlink()
    (run_dir / "frames/f16/seed_0/exact_global/arm_receipt.json").unlink()
    checkpoint_refusal = run_p5_pilot(
        config,
        run_dir,
        "cpu",
        cells=cells,
        corpus_overrides={"replicates": 3},
        model_overrides={"dim": 32},
    )
    assert checkpoint_refusal["complete"] is False
    assert checkpoint_refusal["resumable"] is False
    assert checkpoint_refusal["stopped_for_required_arm_refusal"] is True
    assert "checkpoint identity mismatch" in checkpoint_refusal["required_arm_failure"]["reason"]
    assert "requirements_sha256" in checkpoint_refusal["required_arm_failure"]["reason"]


def test_f64_trainability_null_is_terminal_complete_sealed_and_not_promotable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = [
        P5CellSpec(frames, mechanism)
        for frames in (64, 32)
        for mechanism in ("exact_global", "hierarchical_pooled")
    ]
    config = {
        "profile": "unit-terminal-null",
        "training": {
            "seeds": [0, 1],
            "dense_steps": 1,
            "batch_size": 1,
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
    monkeypatch.setattr(p5_context, "P5_TRAINABILITY_MARGIN", 1.0)
    receipt = run_p5_pilot(
        config,
        tmp_path / "terminal-null",
        "cpu",
        cells=cells,
        corpus_overrides={"replicates": 3, "resolution": 64},
        model_overrides={
            "dim": 8,
            "depth": 1,
            "heads": 1,
            "patch_size": 32,
            "max_resolution": 64,
        },
    )

    assert receipt["trainability_gate"]["outcome"] == "null"
    assert receipt["trainability_gate_failed"] is True
    assert receipt["complete"] is True
    assert receipt["all_ok"] is True
    assert receipt["resumable"] is False
    assert receipt["execution_status"] == "terminal-scientific-null"
    assert receipt["terminal_scientific_stop"] is True
    assert receipt["terminal_stop_reason"] == "f64-trainability-gate-null"
    assert receipt["fresh_challenge_required"] is False
    assert receipt["promotion"]["confirmatory_promotable"] is False
    assert receipt["frames"]["f64"]["seeds_completed"] == 1
    assert receipt["frames"]["f32"]["seeds_completed"] == 1

    without_digest = dict(receipt)
    declared_digest = without_digest.pop("payload_sha256")
    assert json_sha256(without_digest) == declared_digest
    assert receipt["source_bindings"] == [
        {"path": relative, "file_sha256": sha256_file(REPO_ROOT / relative)}
        for relative in p5_context.P5_SOURCE_PATHS
    ]
    assert p5_context.P5_SOURCE_PATHS[-2:] == P5_CORE_RUNTIME_SOURCES
    assert receipt["source_bindings_sha256"] == json_sha256(receipt["source_bindings"])
    expected_requirements = p5_context._checkpoint_requirements_sha256(
        receipt["cell_registry_sha256"], receipt["source_bindings_sha256"]
    )
    assert receipt["checkpoint_requirements_sha256"] == expected_requirements
    for frames in (64, 32):
        cell = json.loads(
            (tmp_path / "terminal-null" / "frames" / f"f{frames}" / "cell_receipt.json").read_text()
        )
        seed = cell["seed_results"]["0"]
        assert seed["source_bindings_sha256"] == receipt["source_bindings_sha256"]
        assert seed["checkpoint_requirements_sha256"] == expected_requirements
        assert all(
            arm["training"]["requirements_sha256"] == expected_requirements
            for arm in seed["mechanisms"].values()
        )

    tampered = copy.deepcopy(receipt)
    tampered["trainability_gate"]["outcome"] = "clears-margin"
    tampered_without_digest = dict(tampered)
    tampered_without_digest.pop("payload_sha256")
    assert json_sha256(tampered_without_digest) != declared_digest


@pytest.mark.parametrize(
    ("wall_budget", "min_free_disk_gb", "expected_status", "wall_stop", "disk_stop"),
    [
        (-1.0, 0.0, "resumable-wall-budget", True, False),
        (600.0, 1.0e12, "resumable-disk-floor", False, True),
    ],
)
def test_p5_resource_stops_are_incomplete_and_resumable(
    tmp_path: Path,
    wall_budget: float,
    min_free_disk_gb: float,
    expected_status: str,
    wall_stop: bool,
    disk_stop: bool,
) -> None:
    config = {
        "profile": "unit-resource-stop",
        "training": {
            "seeds": [0],
            "dense_steps": 1,
            "batch_size": 1,
            "eval_batch_size": 8,
            "learning_rate": 0.0005,
            "weight_decay": 0.02,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "variance_weight": 0.1,
            "checkpoint_every": 1,
            "wall_budget_seconds": wall_budget,
        },
        "screen": {
            "sesoi": 0.10,
            "futility_margin": 0.10,
            "min_free_disk_gb": min_free_disk_gb,
        },
    }
    receipt = run_p5_pilot(
        config,
        tmp_path / expected_status,
        "cpu",
        cells=[P5CellSpec(64, "exact_global")],
        corpus_overrides={"replicates": 3, "resolution": 64},
        model_overrides={
            "dim": 8,
            "depth": 1,
            "heads": 1,
            "patch_size": 32,
            "max_resolution": 64,
        },
        repo_root=tmp_path,
    )

    assert receipt["complete"] is False
    assert receipt["all_ok"] is False
    assert receipt["resumable"] is True
    assert receipt["execution_status"] == expected_status
    assert receipt["terminal_scientific_stop"] is False
    assert receipt["stopped_for_wall_budget"] is wall_stop
    assert receipt["stopped_for_disk_floor"] is disk_stop


def test_fresh_challenge_hint_requires_strict_primary_ci_beyond_sesoi() -> None:
    sesoi = 0.10
    null_rows = {
        "tie_positive": {"n": 5, "lo": sesoi, "hi": 0.30},
        "tie_negative": {"n": 5, "lo": -0.30, "hi": -sesoi},
        "too_few_units": {"n": 1, "lo": 0.20, "hi": 0.30},
        "crosses_boundary": {"n": 5, "lo": 0.05, "hi": 0.30},
    }
    assert _fresh_challenge_required(null_rows, None, sesoi) is False
    assert _fresh_challenge_required(None, null_rows, sesoi) is False
    assert _fresh_challenge_required({}, {}, sesoi) is False

    strict_positive = {"exact_minus_window": {"n": 2, "lo": 0.1000001, "hi": 0.30}}
    strict_negative = {"exact_minus_recurrent": {"n": 3, "lo": -0.40, "hi": -0.1000001}}
    assert _fresh_challenge_required(strict_positive, None, sesoi) is True
    assert _fresh_challenge_required(None, strict_negative, sesoi) is True


@pytest.mark.parametrize(
    ("complete", "all_ok", "resumable", "expected_rc", "published"),
    [
        (True, True, False, 0, True),
        (True, False, False, 1, False),
        (False, False, True, 2, False),
        (False, False, False, 1, False),
    ],
)
def test_p5_cli_publishes_and_succeeds_only_for_complete_all_ok_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete: bool,
    all_ok: bool,
    resumable: bool,
    expected_rc: int,
    published: bool,
) -> None:
    receipt = {
        "complete": complete,
        "all_ok": all_ok,
        "resumable": resumable,
        "execution_status": "complete" if complete and all_ok else "fixture-stop",
        "terminal_scientific_stop": False,
        "terminal_stop_reason": None,
        "resource_telemetry": {"wall_seconds_this_invocation": 0.0},
        "frames": {"f64": {"complete": complete}},
        "trainability_gate_failed": False,
        "promotion": {"confirmatory_promotable": False},
    }
    monkeypatch.setattr(p5_cli, "assert_heavy_lane_free", lambda: None)
    monkeypatch.setattr(p5_cli, "_config", lambda _profile: {"profile": "fixture"})
    monkeypatch.setattr(p5_cli, "run_p5_pilot", lambda *_args: receipt)
    out = tmp_path / "proof.json"

    rc = p5_cli.main(
        [
            "--profile",
            "p5smoke",
            "--device",
            "cpu",
            "--run-dir",
            str(tmp_path / "run"),
            "--out",
            str(out),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert rc == expected_rc
    assert out.exists() is published
    assert summary["proof"] == (str(out) if published else None)
