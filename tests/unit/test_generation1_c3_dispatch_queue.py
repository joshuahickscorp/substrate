from __future__ import annotations

import json
from pathlib import Path

from mop.studies import generation1_c3_dispatch_queue as queue


def test_hawking_mode_is_active_idle_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    core = {"schema": queue.HAWKING_SCHEMA, "active_cells": ["cell"]}
    path.write_text(json.dumps({**core, "state_sha256": queue.canonical_sha256(core)}))
    assert queue.hawking_mode(path) == ("hawking_active", 1, None)
    core["active_cells"] = []
    path.write_text(json.dumps({**core, "state_sha256": queue.canonical_sha256(core)}))
    assert queue.hawking_mode(path) == ("hawking_idle", 8, None)
    path.write_text("{}")
    mode, workers, problem = queue.hawking_mode(path)
    assert mode == "hawking_unknown_fail_closed"
    assert workers == 1
    assert problem is not None


def test_rung_seed_ranges_are_disjoint() -> None:
    configs = [queue.rung_config(index) for index in range(8)]
    train = {
        seed
        for config in configs
        for seed in range(config["train_seed_start"], config["train_seed_start"] + 2)
    }
    heldout = {
        seed
        for config in configs
        for seed in range(config["heldout_seed_start"], config["heldout_seed_start"] + 2)
    }
    assert len(train) == len(heldout) == 16
    assert train.isdisjoint(heldout)


def test_aggregate_is_fail_closed() -> None:
    cells = []
    for _seed in (1, 2):
        cells.append(
            {
                "accuracy": {
                    "learned_dispatch": 0.6,
                    "global_static": 0.5,
                    "difficulty_static": 0.55,
                    "random_actor": 0.4,
                    "context_route_nonpromotable": 0.65,
                    "oracle_nonpromotable": 0.8,
                }
            }
        )
    rung = {
        "result_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "grid": {"train_seed_count": 2, "heldout_seed_count": 2, "completed_cell_count": 2},
        "cells": cells,
    }
    result = queue.aggregate([rung])
    queue.validate_aggregate(result, 1)
    assert result["decision"]["advisory_learned_dispatch_beats_both_static_controls"] is True
    assert result["decision"]["clears_frozen_static_margin_gate"] is True
    assert result["decision"]["ready_for_confirmatory_claim"] is False
    assert result["activation_allowed"] is False
