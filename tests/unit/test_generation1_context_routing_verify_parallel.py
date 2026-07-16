from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mop.studies import generation1_context_routing_verify_parallel as parallel


class _ImmediateExecutor:
    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: object,
        initializer: Any,
        initargs: tuple[object, ...],
    ) -> None:
        del max_workers, mp_context
        initializer(*initargs)

    def submit(self, function: Any, task: parallel.CellTask) -> Future[parallel.CellResult]:
        future: Future[parallel.CellResult] = Future()
        future.set_result(function(task))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures


def test_parallel_worker_reuses_independent_cell_checks(monkeypatch, tmp_path: Path) -> None:
    labels: list[str] = []
    receipt = {"cell_sha256": "a" * 64}
    rebuilt = {"accuracy": {"routed": 0.5}}
    monkeypatch.setattr(parallel, "set_process_label", labels.append)
    monkeypatch.setattr(parallel.torch, "set_num_threads", lambda _threads: None)
    monkeypatch.setattr(parallel.independent, "_read", lambda _path, _label: receipt)
    monkeypatch.setattr(
        parallel.independent,
        "_cell_problems",
        lambda *_args, **_kwargs: ([], rebuilt),
    )
    parallel._initialize_worker({"dataset": {}}, "b" * 64, str(tmp_path))

    result = parallel._verify_cell((7, 2, "proof/cell.json", "a" * 64))

    assert labels == ["mop-c2-verify-worker"]
    assert result == (7, 2, "proof/cell.json", [], rebuilt)


def test_parallel_scheduler_restores_results_by_coordinate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(parallel, "_initialize_worker", lambda *_args: None)
    monkeypatch.setattr(parallel, "_queue_worker_target", lambda _config: (2, "hawking_idle", None))
    monkeypatch.setattr(
        parallel,
        "_verify_cell",
        lambda task: (task[0], task[1], task[2], [], {"coordinate": [task[0], task[1]]}),
    )
    tasks: list[parallel.CellTask] = [
        (11, 1, "proof/b.json", "b" * 64),
        (10, 0, "proof/a.json", "a" * 64),
        (12, 2, "proof/c.json", "c" * 64),
    ]

    results, history = parallel._parallel_reproduce(tasks, {}, "d" * 64, tmp_path)

    assert sorted(results) == [(10, 0), (11, 1), (12, 2)]
    assert results[(10, 0)][4] == {"coordinate": [10, 0]}
    assert history == [
        {
            "mode": "hawking_idle",
            "workers": 2,
            "starting_remaining_cells": 3,
            "ending_remaining_cells": 0,
            "completed_cells": 3,
            "mode_changed": False,
        }
    ]


def test_invalid_inventory_is_bounded_without_spawning_workers() -> None:
    tasks, problems = parallel._inventory_tasks([], [(1, 0), (1, 1)])

    assert tasks == []
    assert problems == ["result cell inventory length drifted"]


def test_parallel_verifier_finalizes_in_canonical_order(monkeypatch, tmp_path: Path) -> None:
    config = {
        "campaign_id": "fixture-c2",
        "seed_start": 10,
        "seed_count": 2,
        "difficulty_separations": [0.5, 1.0],
        "adaptive_resources": {"idle_workers": 25, "hawking_workers": 6},
    }
    coordinates = [(10, 0), (10, 1), (11, 0), (11, 1)]
    inventory = [
        {
            "seed": seed,
            "difficulty_index": difficulty,
            "path": f"cells/{seed}_{difficulty}.json",
            "cell_sha256": f"{index + 1:064x}",
        }
        for index, (seed, difficulty) in enumerate(coordinates)
    ]
    result = {"result_sha256": "a" * 64, "cell_receipts": inventory}
    worker_results = {
        coordinate: (
            coordinate[0],
            coordinate[1],
            inventory[index]["path"],
            [],
            {"order": index},
        )
        for index, coordinate in enumerate(coordinates)
    }
    rebuilt_rows: list[tuple[int, int, Any]] = []

    monkeypatch.setattr(
        parallel,
        "load_config",
        lambda *_args, **_kwargs: (config, "b" * 64, {"verified": True}),
    )
    monkeypatch.setattr(
        parallel.independent,
        "_read",
        lambda _path, label: result if label == "C2 result" else {"clean": True},
    )
    monkeypatch.setattr(
        parallel,
        "_parallel_reproduce",
        lambda *_args, **_kwargs: (
            worker_results,
            [
                {
                    "mode": "hawking_idle",
                    "workers": 25,
                    "starting_remaining_cells": 4,
                    "ending_remaining_cells": 0,
                    "completed_cells": 4,
                    "mode_changed": False,
                }
            ],
        ),
    )

    def rebuild(_config: object, rows: list[tuple[int, int, Any]]) -> dict[str, bool]:
        rebuilt_rows.extend(rows)
        return {"rebuilt": True}

    monkeypatch.setattr(parallel.independent, "_rebuild_aggregate", rebuild)
    monkeypatch.setattr(parallel.independent, "_semantic_problems", lambda *_args: [])
    monkeypatch.setattr(parallel.independent, "_run_canary", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        parallel.independent,
        "_mutation_suite",
        lambda *_args: {"count": 8, "rejected": 8, "all_rejected": True},
    )
    result_path = tmp_path / "proof/result.json"
    out_path = tmp_path / "proof/verification.json"

    verification = parallel.verify_result_parallel(
        tmp_path / "config.json",
        result_path,
        out_path,
        idle_workers=25,
        hawking_workers=6,
        repo_root=tmp_path,
    )

    assert [(seed, difficulty) for seed, difficulty, _ in rebuilt_rows] == coordinates
    assert verification["dataset_reproduction"]["reproduced_cells"] == 4
    assert verification["parallel_execution"]["worker_result_count"] == 4
    assert verification["verification_complete"] is True
    assert json.loads(out_path.read_text(encoding="utf-8")) == verification
    assert verification["verification_sha256"] == parallel.canonical_sha256(
        {key: value for key, value in verification.items() if key != "verification_sha256"}
    )
