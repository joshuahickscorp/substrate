"""Independent verifier for Generation 1 C2 frozen context routing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.studies.generation1_context_routing import (
    ACTORS,
    CELL_SCHEMA,
    CLAIM_SCOPE,
    CONTEXTS,
    RESULT_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_config,
)

_pr1: Any = import_module("scripts.pr1_mode_error_disjointness")
make_dataset = _pr1.make_dataset
run_mode = _pr1.run_mode

VERIFICATION_SCHEMA = "mop-generation1-context-routing-verification/v1"


def _read(path: Path | str, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _seal_ok(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _accuracy(prediction: Sequence[int], truth: Sequence[int], indexes: Sequence[int]) -> float:
    if not indexes:
        return 0.0
    return sum(int(prediction[index] == truth[index]) for index in indexes) / len(indexes)


def _random_actors(effective_seed: int, count: int) -> list[str]:
    generator = torch.Generator().manual_seed(effective_seed + 11_000_019)
    values = torch.randint(0, len(ACTORS), (count,), generator=generator).tolist()
    return [ACTORS[int(value)] for value in values]


def _independent_metrics(
    config: Mapping[str, Any],
    difficulty: int,
    truth: list[int],
    context_ids: list[int],
    predictions: Mapping[str, list[int]],
    random_actor_ids: list[str],
) -> dict[str, Any]:
    routed: list[int] = []
    for index, context_index in enumerate(context_ids):
        context = CONTEXTS[context_index]
        actor = str(config["frozen_route"][context][difficulty])
        routed.append(predictions[actor][index])
    global_static = list(predictions[str(config["controls"]["global_static_actor"])])
    difficulty_static = list(
        predictions[str(config["controls"]["per_difficulty_static_actor"][difficulty])]
    )
    random_prediction = [
        predictions[random_actor_ids[index]][index] for index in range(len(truth))
    ]
    oracle: list[int] = []
    for index, expected in enumerate(truth):
        if any(predictions[actor][index] == expected for actor in ACTORS):
            oracle.append(expected)
        else:
            oracle.append(predictions[ACTORS[0]][index])
    named = {
        "routed": routed,
        "global_static": global_static,
        "difficulty_static": difficulty_static,
        "random_actor": random_prediction,
        "oracle_actor": oracle,
    }
    indexes = list(range(len(truth)))
    accuracy = {
        name: _accuracy(prediction, truth, indexes) for name, prediction in named.items()
    }
    correct = {
        name: sum(int(prediction[index] == truth[index]) for index in indexes)
        for name, prediction in named.items()
    }
    actor_accuracy = {
        actor: _accuracy(predictions[actor], truth, indexes) for actor in ACTORS
    }
    context_accuracy: dict[str, Any] = {}
    for context_index, context in enumerate(CONTEXTS):
        selected = [index for index, value in enumerate(context_ids) if value == context_index]
        context_accuracy[context] = {
            name: _accuracy(prediction, truth, selected) for name, prediction in named.items()
        }
    return {
        "observation_count": len(truth),
        "correct": correct,
        "accuracy": accuracy,
        "actor_accuracy": actor_accuracy,
        "context_accuracy": context_accuracy,
        "routed_minus_global_static": accuracy["routed"] - accuracy["global_static"],
        "routed_minus_difficulty_static": accuracy["routed"] - accuracy["difficulty_static"],
    }


def _interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def _cell_problems(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    seed: int,
    difficulty: int,
    *,
    reproduce_dataset: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    problems: list[str] = []
    prefix = f"seed {seed} difficulty {difficulty}"
    if receipt.get("schema") != CELL_SCHEMA:
        problems.append(f"{prefix}: cell schema drifted")
    if not _seal_ok(receipt, "cell_sha256"):
        problems.append(f"{prefix}: cell seal is invalid")
    expected_identity = (
        receipt.get("campaign_id") == config["campaign_id"]
        and receipt.get("claim_scope") == CLAIM_SCOPE
        and receipt.get("config_file_sha256") == config_sha256
        and receipt.get("seed") == seed
        and receipt.get("difficulty_index") == difficulty
        and receipt.get("complete") is True
    )
    if not expected_identity:
        problems.append(f"{prefix}: cell identity or completion drifted")
    if receipt.get("activation_allowed") is not False or receipt.get("scientific_promotion") is not False:
        problems.append(f"{prefix}: activation or promotion escaped")
    n_test = int(config["dataset"]["n_test"])
    truth = receipt.get("truth")
    context_ids = receipt.get("context_ids")
    predictions = receipt.get("predictions")
    random_actor_ids = receipt.get("random_actor_ids")
    if not isinstance(truth, list) or len(truth) != n_test:
        problems.append(f"{prefix}: truth length drifted")
        return problems, None
    if not isinstance(context_ids, list) or len(context_ids) != n_test or set(context_ids) != set(
        range(len(CONTEXTS))
    ):
        problems.append(f"{prefix}: context inventory drifted")
        return problems, None
    if not isinstance(predictions, dict) or set(predictions) != set(ACTORS) or any(
        not isinstance(predictions[actor], list) or len(predictions[actor]) != n_test
        for actor in ACTORS
    ):
        problems.append(f"{prefix}: prediction inventory drifted")
        return problems, None
    effective_seed = seed + difficulty * 1_000_003
    expected_random = _random_actors(effective_seed, n_test)
    if random_actor_ids != expected_random:
        problems.append(f"{prefix}: random-router authority drifted")
        return problems, None
    typed_truth = [int(value) for value in truth]
    typed_contexts = [int(value) for value in context_ids]
    typed_predictions = {
        actor: [int(value) for value in predictions[actor]] for actor in ACTORS
    }
    rebuilt = _independent_metrics(
        config,
        difficulty,
        typed_truth,
        typed_contexts,
        typed_predictions,
        expected_random,
    )
    if receipt.get("metrics") != rebuilt:
        problems.append(f"{prefix}: raw predictions do not rebuild the declared metrics")
    if reproduce_dataset:
        dataset = config["dataset"]
        xtr, ytr, xte, yte, generated_contexts = make_dataset(
            effective_seed,
            int(dataset["n_train"]),
            int(dataset["n_test"]),
            int(dataset["n_classes"]),
            int(dataset["dim"]),
            float(config["difficulty_separations"][difficulty]),
        )
        if _tensor_digest(xtr, ytr, xte, yte, generated_contexts) != receipt.get(
            "dataset_sha256"
        ):
            problems.append(f"{prefix}: generated dataset digest drifted")
        if [int(value) for value in yte.tolist()] != typed_truth or [
            int(value) for value in generated_contexts.tolist()
        ] != typed_contexts:
            problems.append(f"{prefix}: generated truth or context vector drifted")
    return problems, rebuilt


def _rebuild_aggregate(
    config: Mapping[str, Any],
    rebuilt_rows: Sequence[tuple[int, int, Mapping[str, Any]]],
) -> dict[str, Any]:
    names = ("routed", "global_static", "difficulty_static", "random_actor", "oracle_actor")
    by_seed: dict[int, dict[str, list[float]]] = {}
    by_cell: dict[tuple[str, int], dict[str, list[float]]] = {}
    for seed, difficulty, metrics in rebuilt_rows:
        seed_values = by_seed.setdefault(seed, {name: [] for name in names})
        for name in names:
            seed_values[name].append(float(metrics["accuracy"][name]))
        for context in CONTEXTS:
            values = by_cell.setdefault((context, difficulty), {name: [] for name in names})
            for name in names:
                values[name].append(float(metrics["context_accuracy"][context][name]))
    seed_means = {
        seed: {name: statistics.fmean(values[name]) for name in names}
        for seed, values in by_seed.items()
    }
    overall = {
        name: _interval([values[name] for values in seed_means.values()]) for name in names
    }
    comparisons: dict[str, Any] = {}
    for control in ("global_static", "difficulty_static", "random_actor"):
        differences = [values["routed"] - values[control] for values in seed_means.values()]
        comparisons[f"routed_minus_{control}"] = {
            "interval": _interval(differences),
            "favorable_seed_fraction": sum(value > 0 for value in differences) / len(differences),
        }
    context_difficulty = [
        {
            "context": context,
            "difficulty_index": difficulty,
            "separation": float(config["difficulty_separations"][difficulty]),
            "route_actor": config["frozen_route"][context][difficulty],
            "metrics": {name: _interval(values[name]) for name in names},
        }
        for (context, difficulty), values in sorted(
            by_cell.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]
    criteria = config["criteria"]

    def passes(name: str) -> bool:
        row = comparisons[name]
        interval = row["interval"]
        return bool(
            float(interval["mean"]) >= float(criteria["minimum_mean_advantage"])
            and float(interval["lo"]) > 0
            and float(row["favorable_seed_fraction"])
            >= float(criteria["minimum_favorable_seed_fraction"])
        )

    beats_global = passes("routed_minus_global_static")
    beats_difficulty = passes("routed_minus_difficulty_static")
    confirmed = beats_global and beats_difficulty
    return {
        "overall": overall,
        "comparisons": comparisons,
        "context_difficulty": context_difficulty,
        "decision": {
            "beats_global_static_control": beats_global,
            "beats_per_difficulty_static_control": beats_difficulty,
            "context_labeled_frozen_routing_confirmed": confirmed,
            "ready_to_preregister_g1_c3_learned_dispatch": confirmed,
            "ready_to_train_dispatcher": False,
            "verdict": (
                "context_labeled_frozen_routing_candidate_pending_independent_verification"
                if confirmed
                else "context_labeled_frozen_routing_not_confirmed"
            ),
        },
    }


def _semantic_problems(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    rebuilt: Mapping[str, Any] | None,
) -> list[str]:
    problems: list[str] = []
    if result.get("schema") != RESULT_SCHEMA:
        problems.append("result schema drifted")
    if not _seal_ok(result, "result_sha256"):
        problems.append("result seal is invalid")
    if result.get("campaign_id") != config["campaign_id"] or result.get("claim_scope") != CLAIM_SCOPE:
        problems.append("result campaign or claim scope drifted")
    if result.get("activation_allowed") is not False or result.get("scientific_promotion") is not False:
        problems.append("result activation or promotion escaped")
    if result.get("complete") is not True or result.get("problems") != []:
        problems.append("result is not complete and clean")
    embedded = result.get("config") or {}
    if embedded.get("file_sha256") != config_sha256:
        problems.append("result config binding drifted")
    expected_cells = int(config["seed_count"]) * len(config["difficulty_separations"])
    grid = result.get("grid") or {}
    if (
        grid.get("expected_seed_count") != int(config["seed_count"])
        or grid.get("completed_seed_count") != int(config["seed_count"])
        or grid.get("expected_cell_count") != expected_cells
        or grid.get("completed_cell_count") != expected_cells
    ):
        problems.append("result completion grid drifted")
    inventory = result.get("cell_receipts")
    if not isinstance(inventory, list) or len(inventory) != expected_cells:
        problems.append("result cell inventory drifted")
    if rebuilt is not None:
        for field in ("overall", "comparisons", "context_difficulty", "decision"):
            if result.get(field) != rebuilt[field]:
                problems.append(f"independent {field} rebuild drifted")
    decision = result.get("decision") or {}
    if decision.get("ready_to_train_dispatcher") is not False:
        problems.append("dispatcher training escaped the C2 boundary")
    return problems


def _run_canary(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    seed = int(receipt["seed"])
    difficulty = int(receipt["difficulty_index"])
    effective_seed = seed + difficulty * 1_000_003
    dataset = config["dataset"]
    xtr, ytr, xte, yte, contexts = make_dataset(
        effective_seed,
        int(dataset["n_train"]),
        int(dataset["n_test"]),
        int(dataset["n_classes"]),
        int(dataset["dim"]),
        float(config["difficulty_separations"][difficulty]),
    )
    predictions_match = True
    for actor_index, actor in enumerate(ACTORS):
        prediction = run_mode(
            actor,
            xtr,
            ytr,
            xte,
            int(dataset["n_classes"]),
            int(config["training"]["epochs"]),
            seed=effective_seed + (actor_index + 1) * 100_003,
        )
        predictions_match = predictions_match and [
            int(value) for value in prediction.tolist()
        ] == receipt["predictions"][actor]
    return {
        "seed": seed,
        "difficulty_index": difficulty,
        "dataset_sha256": _tensor_digest(xtr, ytr, xte, yte, contexts),
        "dataset_matches": _tensor_digest(xtr, ytr, xte, yte, contexts)
        == receipt.get("dataset_sha256"),
        "all_actor_predictions_match": predictions_match,
        "passed": predictions_match
        and _tensor_digest(xtr, ytr, xte, yte, contexts) == receipt.get("dataset_sha256"),
    }


def _mutation_suite(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    rebuilt: Mapping[str, Any],
) -> dict[str, Any]:
    mutations: list[tuple[str, dict[str, Any]]] = []

    def reseal(value: dict[str, Any]) -> dict[str, Any]:
        value.pop("result_sha256", None)
        value["result_sha256"] = canonical_sha256(value)
        return value

    for name in (
        "activation_enabled",
        "promotion_enabled",
        "claim_scope_escalated",
        "config_binding_drifted",
        "cell_removed",
        "training_enabled",
        "overall_corrupted",
        "decision_corrupted",
    ):
        value = copy.deepcopy(dict(result))
        if name == "activation_enabled":
            value["activation_allowed"] = True
        elif name == "promotion_enabled":
            value["scientific_promotion"] = True
        elif name == "claim_scope_escalated":
            value["claim_scope"] = "substrate-confirmed"
        elif name == "config_binding_drifted":
            value["config"]["file_sha256"] = "0" * 64
        elif name == "cell_removed":
            value["cell_receipts"].pop()
        elif name == "training_enabled":
            value["decision"]["ready_to_train_dispatcher"] = True
        elif name == "overall_corrupted":
            value["overall"]["routed"]["mean"] += 0.01
        else:
            value["decision"]["context_labeled_frozen_routing_confirmed"] = not value[
                "decision"
            ]["context_labeled_frozen_routing_confirmed"]
        mutations.append((name, reseal(value)))
    rows = []
    for name, value in mutations:
        problems = _semantic_problems(value, config, config_sha256, rebuilt)
        rows.append({"mutation": name, "rejected": bool(problems), "problems": problems})
    return {
        "count": len(rows),
        "rejected": sum(row["rejected"] for row in rows),
        "all_rejected": all(row["rejected"] for row in rows),
        "mutations": rows,
    }


def verify_result(
    config_path: Path | str,
    result_path: Path | str,
    out_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config, config_sha256, prerequisite = load_config(config_path, repo_root=repo_root)
    result = _read(result_path, "C2 result")
    problems: list[str] = []
    inventory = result.get("cell_receipts")
    expected_coordinates = [
        (seed, difficulty)
        for seed in range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
        for difficulty in range(len(config["difficulty_separations"]))
    ]
    rebuilt_rows: list[tuple[int, int, Mapping[str, Any]]] = []
    canary_receipt: dict[str, Any] | None = None
    if not isinstance(inventory, list) or len(inventory) != len(expected_coordinates):
        problems.append("result cell inventory length drifted")
    else:
        observed_coordinates = [
            (row.get("seed"), row.get("difficulty_index")) if isinstance(row, dict) else (None, None)
            for row in inventory
        ]
        if observed_coordinates != expected_coordinates:
            problems.append("result cell inventory order or coordinates drifted")
        for row, (seed, difficulty) in zip(inventory, expected_coordinates, strict=True):
            if not isinstance(row, dict):
                problems.append(f"seed {seed} difficulty {difficulty}: inventory row is invalid")
                continue
            path = (repo_root / str(row.get("path"))).resolve()
            if not path.is_relative_to(repo_root.resolve()):
                problems.append(f"seed {seed} difficulty {difficulty}: receipt path escapes repository")
                continue
            try:
                receipt = _read(path, "C2 cell")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"seed {seed} difficulty {difficulty}: {type(exc).__name__}: {exc}")
                continue
            if row.get("cell_sha256") != receipt.get("cell_sha256"):
                problems.append(f"seed {seed} difficulty {difficulty}: inventory digest drifted")
            cell_problems, metrics = _cell_problems(
                receipt,
                config,
                config_sha256,
                seed,
                difficulty,
                reproduce_dataset=True,
            )
            problems.extend(cell_problems)
            if metrics is not None:
                rebuilt_rows.append((seed, difficulty, metrics))
            if canary_receipt is None and not cell_problems:
                canary_receipt = receipt
    rebuilt = (
        _rebuild_aggregate(config, rebuilt_rows)
        if len(rebuilt_rows) == len(expected_coordinates)
        else None
    )
    problems.extend(_semantic_problems(result, config, config_sha256, rebuilt))
    canary = (
        _run_canary(config, canary_receipt)
        if canary_receipt is not None
        else {"passed": False, "problem": "no clean cell was available for canary replay"}
    )
    if canary.get("passed") is not True:
        problems.append("fresh actor canary replay failed")
    mutation_suite = (
        _mutation_suite(result, config, config_sha256, rebuilt) if rebuilt is not None else None
    )
    if mutation_suite is None or mutation_suite.get("all_rejected") is not True:
        problems.append("mutation suite did not reject every corruption")
    unique_problems = list(dict.fromkeys(problems))
    core = {
        "schema": VERIFICATION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_file_sha256": config_sha256,
        "result_path": str(Path(result_path).resolve().relative_to(repo_root.resolve())),
        "result_sha256": result.get("result_sha256"),
        "prerequisite": prerequisite,
        "independent_recompute": rebuilt,
        "dataset_reproduction": {
            "expected_cells": len(expected_coordinates),
            "reproduced_cells": len(rebuilt_rows),
            "all_dataset_and_metric_reproductions_passed": len(rebuilt_rows)
            == len(expected_coordinates),
        },
        "fresh_actor_canary": canary,
        "mutation_suite": mutation_suite,
        "verification_complete": not unique_problems,
        "problems": unique_problems,
        "interpretation_limit": (
            "Independent verification remains bounded to generated latent data with supplied "
            "diagnostic context labels and grants no learned-dispatch, activation, or substrate claim."
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    verification = {**core, "verification_sha256": canonical_sha256(core)}
    atomic_write_json(out_path, verification)
    return verification


__all__ = ["VERIFICATION_SCHEMA", "verify_result"]
