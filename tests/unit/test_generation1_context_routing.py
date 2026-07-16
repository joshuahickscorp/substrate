from __future__ import annotations

from pathlib import Path

import torch

from mop.config import REPO_ROOT
from mop.studies import generation1_context_routing as routing


def test_official_preregistration_is_frozen_from_verified_c1() -> None:
    config, _, prerequisite = routing.load_config(
        REPO_ROOT / "configs/experiment/generation1_context_routing.json"
    )

    assert config["seed_count"] == 8192
    assert set(range(config["seed_start"], config["seed_start"] + config["seed_count"])).isdisjoint(
        set(range(20260901, 20260949))
    )
    assert prerequisite["c1_independent_verification_complete"] is True
    assert prerequisite["c1_frozen_decisions"] == {
        "frozen_route": config["frozen_route"],
        "global_static_actor": "mlp",
        "per_difficulty_static_actor": ["prototype", "prototype", "knn", "knn", "knn"],
    }
    assert config["adaptive_resources"]["idle_workers"] == 25
    assert config["adaptive_resources"]["hawking_workers"] == 6


def test_cell_receipt_rebuilds_from_raw_predictions(monkeypatch) -> None:
    config, _, _ = routing.load_config(REPO_ROOT / "configs/experiment/generation1_context_routing.json")
    config = {
        **config,
        "dataset": {"n_train": 12, "n_test": 6, "n_classes": 3, "dim": 4},
        "training": {"epochs": 1, "torch_threads": 1},
    }

    def fake_dataset(*_args):
        return (
            torch.zeros((12, 4)),
            torch.tensor([0, 1, 2] * 4),
            torch.zeros((6, 4)),
            torch.tensor([0, 1, 2, 0, 1, 2]),
            torch.tensor([0, 1, 2, 0, 1, 2]),
        )

    actor_to_prediction = {
        actor: torch.tensor([(index + offset) % 3 for index in range(6)])
        for offset, actor in enumerate(routing.ACTORS)
    }

    def fake_mode(actor, *_args, **_kwargs):
        return actor_to_prediction[actor]

    monkeypatch.setattr(routing, "make_dataset", fake_dataset)
    monkeypatch.setattr(routing, "run_mode", fake_mode)
    receipt = routing.run_cell(config, "f" * 64, config["seed_start"], 0)

    routing.validate_cell(receipt, config, "f" * 64, config["seed_start"], 0)
    assert receipt["complete"] is True
    assert receipt["activation_allowed"] is False
    assert receipt["scientific_promotion"] is False
    assert receipt["metrics"]["observation_count"] == 6


def test_queue_target_uses_idle_and_active_sealed_states(tmp_path: Path) -> None:
    config, _, _ = routing.load_config(REPO_ROOT / "configs/experiment/generation1_context_routing.json")
    queue_path = tmp_path / "queue_state.json"
    config = {
        **config,
        "adaptive_resources": {
            **config["adaptive_resources"],
            "hawking_queue_state": str(queue_path),
        },
    }
    core = {
        "schema": routing.HAWKING_QUEUE_SCHEMA,
        "plan_sha256": config["adaptive_resources"]["hawking_plan_sha256"],
        "active_cells": [],
    }
    routing.atomic_write_json(queue_path, {**core, "state_sha256": routing.canonical_sha256(core)})
    assert routing._queue_worker_target(config) == (25, "hawking_idle", None)

    core["active_cells"] = [{"cell_id": "qwen-7b"}]
    routing.atomic_write_json(queue_path, {**core, "state_sha256": routing.canonical_sha256(core)})
    assert routing._queue_worker_target(config) == (6, "hawking_active", None)

    queue_path.write_text("{}", encoding="utf-8")
    workers, mode, problem = routing._queue_worker_target(config)
    assert workers == 6
    assert mode == "hawking_state_fail_closed"
    assert problem is not None


def test_cell_worker_gets_shard_specific_os_label(monkeypatch) -> None:
    labels: list[str] = []
    monkeypatch.setattr(routing, "set_process_label", labels.append)

    routing._label_cell_worker(3)

    assert labels == ["mop-c2-s03-worker"]


def test_aggregate_streams_compact_cell_evidence(monkeypatch, tmp_path: Path) -> None:
    config, config_sha256, prerequisite = routing.load_config(
        REPO_ROOT / "configs/experiment/generation1_context_routing.json"
    )
    config = {
        **config,
        "seed_start": 10,
        "seed_count": 2,
        "shard_count": 1,
        "difficulty_separations": config["difficulty_separations"][:2],
        "frozen_route": {
            context: values[:2] for context, values in config["frozen_route"].items()
        },
        "controls": {
            **config["controls"],
            "per_difficulty_static_actor": config["controls"]["per_difficulty_static_actor"][:2],
        },
    }
    names = ("routed", "global_static", "difficulty_static", "random_actor", "oracle_actor")
    rows = []
    for seed in range(10, 12):
        for difficulty in range(2):
            accuracy = {
                "routed": 0.9,
                "global_static": 0.7,
                "difficulty_static": 0.75,
                "random_actor": 0.6,
                "oracle_actor": 0.95,
            }
            receipt = {
                "cell_sha256": f"{seed * 10 + difficulty:064x}",
                "metrics": {
                    "accuracy": accuracy,
                    "context_accuracy": {
                        context: {name: accuracy[name] for name in names}
                        for context in routing.CONTEXTS
                    },
                },
            }
            rows.append((seed, difficulty, f"cells/{seed}/{difficulty}.json", receipt))

    monkeypatch.setattr(
        routing,
        "load_config",
        lambda *_args, **_kwargs: (config, config_sha256, prerequisite),
    )
    consumed = 0

    def iter_rows(*_args):
        nonlocal consumed
        for row in rows:
            consumed += 1
            yield row

    monkeypatch.setattr(routing, "_iter_all_cells", iter_rows)
    result = routing.aggregate_result(tmp_path / "config.json", tmp_path / "work", tmp_path / "out.json")

    assert consumed == 4
    assert result["grid"] == {
        "expected_seed_count": 2,
        "completed_seed_count": 2,
        "expected_cell_count": 4,
        "completed_cell_count": 4,
    }
    assert len(result["cell_receipts"]) == 4
    assert all(set(row) == {"path", "seed", "difficulty_index", "cell_sha256"} for row in result["cell_receipts"])
