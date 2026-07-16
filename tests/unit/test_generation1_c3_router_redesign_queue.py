from __future__ import annotations

from mop.studies import generation1_c3_router_redesign_queue as queue


def test_rung_seed_ranges_are_disjoint_and_fresh() -> None:
    configs = [queue.rung_config(index) for index in range(queue.DEFAULT_RUNG_COUNT)]
    train = {
        seed
        for config in configs
        for seed in range(config["train_seed_start"], config["train_seed_start"] + queue.SEEDS_PER_RUNG)
    }
    heldout = {
        seed
        for config in configs
        for seed in range(config["heldout_seed_start"], config["heldout_seed_start"] + queue.SEEDS_PER_RUNG)
    }
    assert len(train) == len(heldout) == queue.DEFAULT_RUNG_COUNT * queue.SEEDS_PER_RUNG
    assert train.isdisjoint(heldout)
    assert min(train) > 20_280_000


def test_aggregate_is_sealed_and_never_confirmatory() -> None:
    variants = [{"variant_id": "v1"}, {"variant_id": "v2"}]
    cells = [
        {
            "variant_accuracy": {"v1": 0.62, "v2": 0.58},
            "control_accuracy": {
                "global_static": 0.50,
                "difficulty_static": 0.54,
                "random_actor": 0.40,
                "context_route_nonpromotable": 0.65,
                "oracle_nonpromotable": 0.80,
            },
        },
        {
            "variant_accuracy": {"v1": 0.64, "v2": 0.57},
            "control_accuracy": {
                "global_static": 0.51,
                "difficulty_static": 0.55,
                "random_actor": 0.41,
                "context_route_nonpromotable": 0.66,
                "oracle_nonpromotable": 0.81,
            },
        },
    ]
    rung = {
        "result_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "config": {"variants": variants},
        "grid": {"train_seed_count": 3, "heldout_seed_count": 3, "completed_cell_count": 2},
        "cells": cells,
    }
    result = queue.aggregate([rung])
    queue.validate_aggregate(result, 1)
    assert result["best_exploratory_variant"] == "v1"
    assert result["decision"]["best_variant_clears_gate"] is True
    assert result["decision"]["ready_for_confirmatory_claim"] is False
    assert result["activation_allowed"] is False
