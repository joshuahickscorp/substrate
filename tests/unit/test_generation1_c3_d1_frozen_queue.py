from __future__ import annotations

from mop.studies import generation1_c3_d1_frozen_queue as queue


def test_screen_binding_and_frozen_variant_are_exact() -> None:
    binding = queue.screen_binding()
    variant = queue.frozen_variant()
    assert binding["frozen_variant_id"] == "centroid-h64-e60-lr03"
    assert variant == {
        "variant_id": "centroid-h64-e60-lr03",
        "feature_set": "centroid",
        "hidden": 64,
        "epochs": 60,
        "lr": 0.003,
    }


def test_all_frozen_seed_partitions_are_disjoint() -> None:
    intervals = []
    for index in range(queue.DEFAULT_RUNG_COUNT):
        config = queue.rung_config(index)
        intervals.extend(
            (
                config["train_seed_start"],
                config["train_seed_start"] + config["train_seed_count"],
                f"{index}:train",
            )
            for _ in (0,)
        )
        intervals.extend(
            (
                config["heldout_seed_start"],
                config["heldout_seed_start"] + config["heldout_seed_count"],
                f"{index}:heldout",
            )
            for _ in (0,)
        )
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:], strict=False):
        assert previous[1] <= current[0], (previous, current)
    assert intervals[0][0] > 20_319_216


def _fake_rung(*, learned: float, global_static: float, difficulty_static: float, context: float):
    cells = []
    for seed in (1, 2, 3, 4):
        for difficulty in range(5):
            cells.append(
                {
                    "seed": seed,
                    "difficulty_index": difficulty,
                    "variant_accuracy": {queue.FROZEN_VARIANT_ID: learned},
                    "control_accuracy": {
                        "global_static": global_static,
                        "difficulty_static": difficulty_static,
                        "random_actor": global_static,
                        "context_route_nonpromotable": context,
                        "oracle_nonpromotable": 0.8,
                    },
                }
            )
    return {
        "result_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "grid": {"train_seed_count": 4, "heldout_seed_count": 4, "completed_cell_count": 20},
        "cells": cells,
    }


def test_phase_summary_applies_every_frozen_gate() -> None:
    passing = queue.summarize_phase(
        [_fake_rung(learned=0.63, global_static=0.55, difficulty_static=0.58, context=0.64)]
    )
    assert passing["all_frozen_criteria_passed"] is True
    assert all(passing["conditions"].values())

    context_failure = queue.summarize_phase(
        [_fake_rung(learned=0.60, global_static=0.55, difficulty_static=0.58, context=0.64)]
    )
    assert context_failure["conditions"]["context_route_gap_gate"] is False
    assert context_failure["all_frozen_criteria_passed"] is False


def test_status_exposes_both_phase_progress_and_eta(tmp_path) -> None:
    root = tmp_path
    status = queue._status(
        root=root,
        rung_count=queue.DEFAULT_RUNG_COUNT,
        receipts={},
        attempts={},
        durations={0: 120.0},
        state="running",
        mode="hawking_active",
        workers=1,
        problem=None,
    )
    assert status["phase_progress"]["producer"] == {"complete": 0, "total": 288}
    assert status["phase_progress"]["challenge"] == {"complete": 0, "total": 288}
    assert status["adaptive_execution"]["eta_seconds"] == 120.0 * 576
    assert status["activation_allowed"] is False
