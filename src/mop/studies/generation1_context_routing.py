
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import statistics
import tempfile
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import suppress
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.substrate.events import canonical_bytes, canonical_sha256

_pr1: Any = import_module("scripts.pr1_mode_error_disjointness")
ACTORS = tuple(str(value) for value in _pr1.MODES)
CONTEXTS = tuple(str(value) for value in _pr1.SUBPOPS)
make_dataset = _pr1.make_dataset
run_mode = _pr1.run_mode

CONFIG_SCHEMA = "mop-generation1-context-routing-config/v1"
CELL_SCHEMA = "mop-generation1-context-routing-cell/v1"
SHARD_SCHEMA = "mop-generation1-context-routing-shard/v1"
RESULT_SCHEMA = "mop-generation1-context-routing/v1"
CLAIM_SCOPE = "generated-context-labeled-frozen-routing-confirmation-only"
TASK_FAMILY = "generated-mixed-multiclass-latents"
MAX_IDLE_WORKERS = 25
HAWKING_QUEUE_SCHEMA = "hawking.doctor_v5_ultra_queue_state.v1"


def _label_cell_worker(shard_index: int) -> None:
    set_process_label(f"mop-c2-s{shard_index:02d}-worker")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = {key: item for key, item in value.items() if key != field}
    return {**core, field: canonical_sha256(core)}


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if value.get(field) != expected:
        raise ValueError(f"{label} self-seal is invalid")


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path | str, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _atlas_frozen_decisions(atlas: Mapping[str, Any]) -> dict[str, Any]:
    rows = atlas.get("competence_tensor")
    if not isinstance(rows, list) or not rows:
        raise ValueError("C1 competence tensor is unavailable")
    by_cell: dict[tuple[str, int], list[tuple[float, str]]] = {}
    actor_values: dict[str, list[float]] = {actor: [] for actor in ACTORS}
    difficulty_values: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("C1 competence tensor row is invalid")
        actor = str(row.get("actor"))
        context = str(row.get("context"))
        difficulty = int(row.get("difficulty_index", -1))
        if actor not in ACTORS or context not in CONTEXTS or difficulty < 0:
            raise ValueError("C1 competence tensor coordinate drifted")
        quality = row.get("quality") or {}
        mean_value = quality.get("mean")
        if not isinstance(mean_value, int | float) or isinstance(mean_value, bool):
            raise ValueError("C1 competence tensor mean is invalid")
        mean = float(mean_value)
        if not math.isfinite(mean):
            raise ValueError("C1 competence tensor contains a non-finite mean")
        by_cell.setdefault((context, difficulty), []).append((mean, actor))
        actor_values[actor].append(mean)
        difficulty_values.setdefault(difficulty, {}).setdefault(actor, []).append(mean)
    difficulty_count = len(atlas.get("config", {}).get("difficulty_separations", []))
    expected_cells = {(context, difficulty) for context in CONTEXTS for difficulty in range(difficulty_count)}
    if set(by_cell) != expected_cells or any(len(values) != len(ACTORS) for values in by_cell.values()):
        raise ValueError("C1 competence tensor is not a complete context-difficulty-actor grid")
    route = {
        context: [max(by_cell[(context, difficulty)])[1] for difficulty in range(difficulty_count)]
        for context in CONTEXTS
    }
    global_actor = max((statistics.fmean(actor_values[actor]), actor) for actor in ACTORS)[1]
    per_difficulty = [
        max((statistics.fmean(difficulty_values[difficulty][actor]), actor) for actor in ACTORS)[1]
        for difficulty in range(difficulty_count)
    ]
    return {
        "frozen_route": route,
        "global_static_actor": global_actor,
        "per_difficulty_static_actor": per_difficulty,
    }


def load_config(
    path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    source = Path(path).resolve()
    config = _read_json(source, "C2 context-routing config")
    _exact_keys(
        config,
        {
            "schema",
            "campaign_id",
            "claim_scope",
            "prerequisite",
            "seed_start",
            "seed_count",
            "shard_count",
            "difficulty_separations",
            "dataset",
            "training",
            "frozen_route",
            "controls",
            "criteria",
            "adaptive_resources",
            "activation_allowed",
            "scientific_promotion",
        },
        "C2 context-routing config",
    )
    if config["schema"] != CONFIG_SCHEMA or config["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("C2 config schema or claim scope drifted")
    if config["activation_allowed"] is not False or config["scientific_promotion"] is not False:
        raise ValueError("C2 config activation or scientific promotion escaped")
    for field in ("seed_start", "seed_count", "shard_count"):
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError(f"C2 {field} must be a positive integer")
    if config["seed_count"] < 32 or config["seed_count"] % config["shard_count"]:
        raise ValueError("C2 requires at least 32 seeds evenly divisible across shards")
    separations = config["difficulty_separations"]
    if not isinstance(separations, list) or len(separations) < 3:
        raise ValueError("C2 difficulty grid is too small")
    dataset = config["dataset"]
    _exact_keys(dataset, {"n_train", "n_test", "n_classes", "dim"}, "C2 dataset")
    if any(isinstance(dataset[key], bool) or int(dataset[key]) <= 0 for key in dataset):
        raise ValueError("C2 dataset dimensions must be positive integers")
    training = config["training"]
    _exact_keys(training, {"epochs", "torch_threads"}, "C2 training")
    if int(training["epochs"]) <= 0 or int(training["torch_threads"]) != 1:
        raise ValueError("C2 requires positive epochs and exactly one thread per actor")
    route = config["frozen_route"]
    if set(route) != set(CONTEXTS) or any(
        not isinstance(route[context], list)
        or len(route[context]) != len(separations)
        or any(actor not in ACTORS for actor in route[context])
        for context in CONTEXTS
    ):
        raise ValueError("C2 frozen route inventory drifted")
    controls = config["controls"]
    _exact_keys(
        controls,
        {
            "global_static_actor",
            "per_difficulty_static_actor",
            "deterministic_random_actor_router",
            "per_example_oracle_actor",
        },
        "C2 controls",
    )
    if controls["global_static_actor"] not in ACTORS or any(
        actor not in ACTORS for actor in controls["per_difficulty_static_actor"]
    ):
        raise ValueError("C2 static control actor drifted")
    if len(controls["per_difficulty_static_actor"]) != len(separations):
        raise ValueError("C2 per-difficulty control length drifted")
    if (
        controls["deterministic_random_actor_router"] is not True
        or controls["per_example_oracle_actor"] is not True
    ):
        raise ValueError("C2 controls must remain enabled")
    criteria = config["criteria"]
    _exact_keys(
        criteria,
        {
            "minimum_mean_advantage",
            "minimum_favorable_seed_fraction",
            "require_95ci_lower_above_zero",
        },
        "C2 criteria",
    )
    if (
        not 0 <= float(criteria["minimum_mean_advantage"]) <= 1
        or not 0 <= float(criteria["minimum_favorable_seed_fraction"]) <= 1
    ):
        raise ValueError("C2 criteria must lie in [0, 1]")
    if criteria["require_95ci_lower_above_zero"] is not True:
        raise ValueError("C2 confidence-bound criterion may not be disabled")
    resources = config["adaptive_resources"]
    _exact_keys(
        resources,
        {"idle_workers", "hawking_workers", "hawking_queue_state", "hawking_plan_sha256"},
        "C2 adaptive resources",
    )
    if (
        resources["idle_workers"] != MAX_IDLE_WORKERS
        or not 1 <= int(resources["hawking_workers"]) < MAX_IDLE_WORKERS
    ):
        raise ValueError("C2 adaptive worker envelope drifted")

    binding = config["prerequisite"]
    _exact_keys(
        binding,
        {
            "atlas_path",
            "atlas_file_sha256",
            "atlas_sha256",
            "verification_path",
            "verification_file_sha256",
            "verification_sha256",
        },
        "C2 prerequisite binding",
    )
    atlas_path = (repo_root / str(binding["atlas_path"])).resolve()
    verification_path = (repo_root / str(binding["verification_path"])).resolve()
    if not atlas_path.is_relative_to(repo_root.resolve()) or not verification_path.is_relative_to(
        repo_root.resolve()
    ):
        raise ValueError("C2 prerequisite path escapes the repository")
    atlas = _read_json(atlas_path, "C1 atlas")
    verification = _read_json(verification_path, "C1 verification")
    if (
        sha256_file(atlas_path) != binding["atlas_file_sha256"]
        or atlas.get("atlas_sha256") != binding["atlas_sha256"]
    ):
        raise ValueError("C1 atlas authority drifted")
    if (
        sha256_file(verification_path) != binding["verification_file_sha256"]
        or verification.get("verification_sha256") != binding["verification_sha256"]
    ):
        raise ValueError("C1 verification authority drifted")
    _validate_seal(atlas, "atlas_sha256", "C1 atlas")
    _validate_seal(verification, "verification_sha256", "C1 verification")
    if verification.get("verification_complete") is not True or verification.get("problems") != []:
        raise ValueError("C1 independent verification is not clean")
    mutation = verification.get("mutation_suite") or {}
    if mutation.get("all_rejected") is not True or int(mutation.get("rejected", -1)) != int(
        mutation.get("count", -2)
    ):
        raise ValueError("C1 mutation suite is not clean")
    decision = atlas.get("decision") or {}
    if (
        decision.get("ready_to_preregister_g1_c2") is not True
        or decision.get("ready_to_train_dispatcher") is not False
    ):
        raise ValueError("C1 does not license this bounded C2 preregistration")
    c1_config = atlas.get("config") or {}
    if (
        c1_config.get("actors") != list(ACTORS)
        or c1_config.get("contexts") != list(CONTEXTS)
        or c1_config.get("difficulty_separations") != [float(value) for value in separations]
        or c1_config.get("dataset") != dataset
        or int((c1_config.get("training") or {}).get("epochs", -1)) != int(training["epochs"])
    ):
        raise ValueError("C2 generated bed drifted from C1")
    c1_seeds = set(int(value) for value in c1_config.get("seeds", []))
    c2_seeds = set(range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"])))
    if c1_seeds & c2_seeds:
        raise ValueError("C2 fresh seeds overlap C1")
    frozen = _atlas_frozen_decisions(atlas)
    if frozen["frozen_route"] != route:
        raise ValueError("C2 route is not the exact C1 cell-mean winner table")
    if frozen["global_static_actor"] != controls["global_static_actor"]:
        raise ValueError("C2 global static control is not frozen from C1")
    if frozen["per_difficulty_static_actor"] != controls["per_difficulty_static_actor"]:
        raise ValueError("C2 per-difficulty static control is not frozen from C1")
    prerequisite = {
        **dict(binding),
        "c1_frozen_decisions": frozen,
        "c1_independent_verification_complete": True,
    }
    return config, sha256_file(source), prerequisite


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _accuracy(prediction: Sequence[int], truth: Sequence[int], indexes: Sequence[int]) -> float:
    return sum(int(prediction[index] == truth[index]) for index in indexes) / len(indexes) if indexes else 0.0


def _random_actor_ids(effective_seed: int, count: int) -> list[str]:
    generator = torch.Generator().manual_seed(effective_seed + 11_000_019)
    indexes = torch.randint(0, len(ACTORS), (count,), generator=generator).tolist()
    return [ACTORS[int(index)] for index in indexes]


def _cell_metrics(
    config: Mapping[str, Any],
    difficulty_index: int,
    truth: list[int],
    context_ids: list[int],
    predictions: Mapping[str, list[int]],
    random_actor_ids: list[str],
) -> dict[str, Any]:
    route = config["frozen_route"]
    controls = config["controls"]
    routed = [
        predictions[str(route[CONTEXTS[context]][difficulty_index])][index]
        for index, context in enumerate(context_ids)
    ]
    global_static = list(predictions[str(controls["global_static_actor"])])
    difficulty_static = list(predictions[str(controls["per_difficulty_static_actor"][difficulty_index])])
    random_prediction = [predictions[random_actor_ids[index]][index] for index in range(len(truth))]
    oracle = [
        truth[index]
        if any(predictions[actor][index] == truth[index] for actor in ACTORS)
        else predictions[ACTORS[0]][index]
        for index in range(len(truth))
    ]
    named = {
        "routed": routed,
        "global_static": global_static,
        "difficulty_static": difficulty_static,
        "random_actor": random_prediction,
        "oracle_actor": oracle,
    }
    all_indexes = list(range(len(truth)))
    accuracy = {name: _accuracy(prediction, truth, all_indexes) for name, prediction in named.items()}
    correct = {
        name: sum(int(prediction[index] == truth[index]) for index in all_indexes)
        for name, prediction in named.items()
    }
    actor_accuracy = {actor: _accuracy(predictions[actor], truth, all_indexes) for actor in ACTORS}
    context_accuracy: dict[str, Any] = {}
    for context_index, context in enumerate(CONTEXTS):
        indexes = [index for index, value in enumerate(context_ids) if value == context_index]
        context_accuracy[context] = {
            name: _accuracy(prediction, truth, indexes) for name, prediction in named.items()
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


def run_cell(
    config: Mapping[str, Any],
    config_sha256: str,
    seed: int,
    difficulty_index: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    started = time.perf_counter()
    separation = float(config["difficulty_separations"][difficulty_index])
    effective_seed = seed + difficulty_index * 1_000_003
    dataset = config["dataset"]
    xtr, ytr, xte, yte, context_tensor = make_dataset(
        effective_seed,
        int(dataset["n_train"]),
        int(dataset["n_test"]),
        int(dataset["n_classes"]),
        int(dataset["dim"]),
        separation,
    )
    predictions: dict[str, list[int]] = {}
    actor_seconds: dict[str, float] = {}
    for actor_index, actor in enumerate(ACTORS):
        actor_started = time.perf_counter()
        prediction = run_mode(
            actor,
            xtr,
            ytr,
            xte,
            int(dataset["n_classes"]),
            int(config["training"]["epochs"]),
            seed=effective_seed + (actor_index + 1) * 100_003,
        )
        actor_seconds[actor] = time.perf_counter() - actor_started
        predictions[actor] = [int(value) for value in prediction.tolist()]
    truth = [int(value) for value in yte.tolist()]
    context_ids = [int(value) for value in context_tensor.tolist()]
    random_actor_ids = _random_actor_ids(effective_seed, len(truth))
    core = {
        "schema": CELL_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_file_sha256": config_sha256,
        "seed": seed,
        "difficulty_index": difficulty_index,
        "separation": separation,
        "effective_seed": effective_seed,
        "dataset_sha256": _tensor_sha256(xtr, ytr, xte, yte, context_tensor),
        "truth": truth,
        "context_ids": context_ids,
        "predictions": predictions,
        "random_actor_ids": random_actor_ids,
        "metrics": _cell_metrics(
            config,
            difficulty_index,
            truth,
            context_ids,
            predictions,
            random_actor_ids,
        ),
        "actor_wall_seconds": actor_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "complete": True,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "cell_sha256")


def validate_cell(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    seed: int,
    difficulty_index: int,
) -> None:
    if receipt.get("schema") != CELL_SCHEMA:
        raise ValueError("C2 cell schema drifted")
    _validate_seal(receipt, "cell_sha256", "C2 cell")
    if (
        receipt.get("campaign_id") != config["campaign_id"]
        or receipt.get("claim_scope") != CLAIM_SCOPE
        or receipt.get("config_file_sha256") != config_sha256
        or receipt.get("seed") != seed
        or receipt.get("difficulty_index") != difficulty_index
        or receipt.get("complete") is not True
    ):
        raise ValueError("C2 cell identity or completion drifted")
    if receipt.get("activation_allowed") is not False or receipt.get("scientific_promotion") is not False:
        raise ValueError("C2 cell activation or promotion escaped")
    if not math.isclose(
        float(receipt.get("separation", -1)),
        float(config["difficulty_separations"][difficulty_index]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("C2 cell separation drifted")
    expected_effective = seed + difficulty_index * 1_000_003
    if receipt.get("effective_seed") != expected_effective:
        raise ValueError("C2 cell effective seed drifted")
    n_test = int(config["dataset"]["n_test"])
    truth = receipt.get("truth")
    context_ids = receipt.get("context_ids")
    predictions = receipt.get("predictions")
    random_actor_ids = receipt.get("random_actor_ids")
    if not isinstance(truth, list) or len(truth) != n_test:
        raise ValueError("C2 cell truth length drifted")
    if (
        not isinstance(context_ids, list)
        or len(context_ids) != n_test
        or set(context_ids) != set(range(len(CONTEXTS)))
    ):
        raise ValueError("C2 cell context inventory drifted")
    if (
        not isinstance(predictions, dict)
        or set(predictions) != set(ACTORS)
        or any(
            not isinstance(predictions[actor], list) or len(predictions[actor]) != n_test for actor in ACTORS
        )
    ):
        raise ValueError("C2 cell actor prediction inventory drifted")
    if random_actor_ids != _random_actor_ids(expected_effective, n_test):
        raise ValueError("C2 deterministic random-router authority drifted")
    rebuilt = _cell_metrics(
        config,
        difficulty_index,
        [int(value) for value in truth],
        [int(value) for value in context_ids],
        {actor: [int(value) for value in predictions[actor]] for actor in ACTORS},
        [str(value) for value in random_actor_ids],
    )
    if receipt.get("metrics") != rebuilt:
        raise ValueError("C2 cell metrics do not rebuild from raw predictions")


def cell_path(work_root: Path | str, seed: int, difficulty_index: int) -> Path:
    return Path(work_root) / "cells" / f"seed_{seed}" / f"difficulty_{difficulty_index}.json"


def _queue_worker_target(config: Mapping[str, Any]) -> tuple[int, str, str | None]:
    resources = config["adaptive_resources"]
    idle_workers = int(resources["idle_workers"])
    hawking_workers = int(resources["hawking_workers"])
    path = Path(str(resources["hawking_queue_state"]))
    try:
        state = _read_json(path, "Hawking queue state")
        _validate_seal(state, "state_sha256", "Hawking queue state")
        if state.get("schema") != HAWKING_QUEUE_SCHEMA:
            raise ValueError("Hawking queue schema drifted")
        if state.get("plan_sha256") != resources["hawking_plan_sha256"]:
            raise ValueError("Hawking plan authority drifted")
        active = state.get("active_cells")
        if not isinstance(active, list):
            raise ValueError("Hawking active-cell inventory is invalid")
        if active:
            return hawking_workers, "hawking_active", None
        return idle_workers, "hawking_idle", None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return hawking_workers, "hawking_state_fail_closed", f"{type(exc).__name__}: {exc}"


def _interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def _shard_report(
    config: Mapping[str, Any],
    config_sha256: str,
    shard_index: int,
    work_root: Path,
    completed: Sequence[tuple[int, int, Mapping[str, Any]]],
    mode_history: Sequence[Mapping[str, Any]],
    failures: Mapping[str, int],
) -> dict[str, Any]:
    expected_seed_count = int(config["seed_count"]) // int(config["shard_count"])
    expected_cells = expected_seed_count * len(config["difficulty_separations"])
    complete = len(completed) == expected_cells
    core = {
        "schema": SHARD_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_file_sha256": config_sha256,
        "shard_index": shard_index,
        "shard_count": int(config["shard_count"]),
        "grid": {
            "expected_seed_count": expected_seed_count,
            "expected_cell_count": expected_cells,
            "completed_cell_count": len(completed),
        },
        "cell_receipts": [
            {
                "path": str(
                    cell_path(work_root, seed, difficulty).resolve().relative_to(REPO_ROOT.resolve())
                ),
                "seed": seed,
                "difficulty_index": difficulty,
                "cell_sha256": receipt["cell_sha256"],
            }
            for seed, difficulty, receipt in completed
        ],
        "adaptive_execution": {
            "idle_workers": int(config["adaptive_resources"]["idle_workers"]),
            "hawking_workers": int(config["adaptive_resources"]["hawking_workers"]),
            "mode_history": list(mode_history),
            "retried_cell_failures": dict(sorted(failures.items())),
        },
        "complete": complete,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "shard_sha256")


def run_shard(
    config_path: Path | str,
    work_root: Path | str,
    out_path: Path | str,
    shard_index: int,
    *,
    repo_root: Path = REPO_ROOT,
    max_new_cells: int | None = None,
) -> dict[str, Any]:
    config, config_sha256, _ = load_config(config_path, repo_root=repo_root)
    shard_count = int(config["shard_count"])
    if isinstance(shard_index, bool) or not 0 <= shard_index < shard_count:
        raise ValueError("C2 shard index is outside the configured range")
    if max_new_cells is not None and (
        isinstance(max_new_cells, bool) or not isinstance(max_new_cells, int) or max_new_cells < 0
    ):
        raise ValueError("max_new_cells must be a nonnegative integer or null")
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [
        seed
        for seed in range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
        if (seed - int(config["seed_start"])) % shard_count == shard_index
    ]
    coordinates = [
        (seed, difficulty) for seed in seeds for difficulty in range(len(config["difficulty_separations"]))
    ]
    receipts: dict[tuple[int, int], dict[str, Any]] = {}
    for seed, difficulty in coordinates:
        path = cell_path(root, seed, difficulty)
        if not path.is_file():
            continue
        try:
            receipt = _read_json(path, "C2 cell")
            validate_cell(receipt, config, config_sha256, seed, difficulty)
            receipts[(seed, difficulty)] = receipt
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    pending = [coordinate for coordinate in coordinates if coordinate not in receipts]
    if max_new_cells is not None:
        pending = pending[:max_new_cells]
    mode_history: list[dict[str, Any]] = []
    failures: dict[str, int] = {}

    def publish() -> dict[str, Any]:
        completed = [
            (seed, difficulty, receipts[(seed, difficulty)])
            for seed, difficulty in coordinates
            if (seed, difficulty) in receipts
        ]
        report = _shard_report(
            config,
            config_sha256,
            shard_index,
            root,
            completed,
            mode_history,
            failures,
        )
        atomic_write_json(out_path, report)
        return report

    publish()
    retry_queue = deque(pending)
    while retry_queue:
        workers, mode, problem = _queue_worker_target(config)
        entry: dict[str, Any] = {
            "mode": mode,
            "workers": workers,
            "started_at_unix": time.time(),
            "starting_remaining_cells": len(retry_queue),
        }
        if problem is not None:
            entry["state_problem"] = problem
        mode_history.append(entry)
        completed_in_mode = 0
        since_publish = 0
        mode_changed = False
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_label_cell_worker,
            initargs=(shard_index,),
        ) as executor:
            active: dict[Any, tuple[int, int]] = {}
            while retry_queue or active:
                if not mode_changed:
                    observed_workers, observed_mode, _ = _queue_worker_target(config)
                    mode_changed = observed_workers != workers or observed_mode != mode
                while not mode_changed and retry_queue and len(active) < workers:
                    seed, difficulty = retry_queue.popleft()
                    future = executor.submit(
                        run_cell,
                        config,
                        config_sha256,
                        seed,
                        difficulty,
                    )
                    active[future] = (seed, difficulty)
                if not active:
                    break
                done, _ = wait(tuple(active), timeout=2.0, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    seed, difficulty = active.pop(future)
                    key = f"{seed}:{difficulty}"
                    try:
                        receipt = future.result()
                        validate_cell(receipt, config, config_sha256, seed, difficulty)
                        atomic_write_json(cell_path(root, seed, difficulty), receipt)
                        receipts[(seed, difficulty)] = receipt
                    except Exception:
                        failures[key] = failures.get(key, 0) + 1
                        if failures[key] >= 3:
                            publish()
                            raise
                        retry_queue.append((seed, difficulty))
                    completed_in_mode += 1
                    since_publish += 1
                if since_publish >= workers:
                    publish()
                    since_publish = 0
            if since_publish:
                publish()
        entry["finished_at_unix"] = time.time()
        entry["ending_remaining_cells"] = len(retry_queue)
        entry["completed_cells"] = completed_in_mode
        entry["mode_changed"] = mode_changed
    return publish()


def _iter_all_cells(
    config: Mapping[str, Any],
    config_sha256: str,
    work_root: Path,
) -> Iterator[tuple[int, int, str, dict[str, Any]]]:
    for seed in range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"])):
        for difficulty in range(len(config["difficulty_separations"])):
            path = cell_path(work_root, seed, difficulty)
            receipt = _read_json(path, "C2 cell")
            validate_cell(receipt, config, config_sha256, seed, difficulty)
            yield (
                seed,
                difficulty,
                str(path.resolve().relative_to(REPO_ROOT.resolve())),
                receipt,
            )


def aggregate_result(
    config_path: Path | str,
    work_root: Path | str,
    out_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config, config_sha256, prerequisite = load_config(config_path, repo_root=repo_root)
    names = ("routed", "global_static", "difficulty_static", "random_actor", "oracle_actor")
    by_seed: dict[int, dict[str, list[float]]] = {}
    by_cell: dict[tuple[str, int], dict[str, list[float]]] = {}
    cell_receipts: list[dict[str, Any]] = []
    completed_cell_count = 0
    for seed, difficulty, path, receipt in _iter_all_cells(config, config_sha256, Path(work_root)):
        metrics = receipt["metrics"]
        seed_values = by_seed.setdefault(seed, {name: [] for name in names})
        for name in names:
            seed_values[name].append(float(metrics["accuracy"][name]))
        for context in CONTEXTS:
            values = by_cell.setdefault((context, difficulty), {name: [] for name in names})
            for name in names:
                values[name].append(float(metrics["context_accuracy"][context][name]))
        cell_receipts.append(
            {
                "path": path,
                "seed": seed,
                "difficulty_index": difficulty,
                "cell_sha256": receipt["cell_sha256"],
            }
        )
        completed_cell_count += 1
    seed_means = {
        seed: {name: statistics.fmean(values[name]) for name in names} for seed, values in by_seed.items()
    }
    overall = {name: _interval([values[name] for values in seed_means.values()]) for name in names}
    comparisons: dict[str, Any] = {}
    for control in ("global_static", "difficulty_static", "random_actor"):
        deltas = [values["routed"] - values[control] for values in seed_means.values()]
        comparisons[f"routed_minus_{control}"] = {
            "interval": _interval(deltas),
            "favorable_seed_fraction": sum(value > 0 for value in deltas) / len(deltas),
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
            and float(row["favorable_seed_fraction"]) >= float(criteria["minimum_favorable_seed_fraction"])
        )

    beats_global = passes("routed_minus_global_static")
    beats_difficulty = passes("routed_minus_difficulty_static")
    confirmed = beats_global and beats_difficulty
    core = {
        "schema": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "task_family": TASK_FAMILY,
        "prerequisite": prerequisite,
        "config": {
            "file_sha256": config_sha256,
            "seed_start": int(config["seed_start"]),
            "seed_count": int(config["seed_count"]),
            "shard_count": int(config["shard_count"]),
            "difficulty_separations": [float(value) for value in config["difficulty_separations"]],
            "dataset": dict(config["dataset"]),
            "training": dict(config["training"]),
            "frozen_route": dict(config["frozen_route"]),
            "controls": dict(config["controls"]),
            "criteria": dict(config["criteria"]),
            "adaptive_resources": dict(config["adaptive_resources"]),
        },
        "grid": {
            "expected_seed_count": int(config["seed_count"]),
            "completed_seed_count": len(by_seed),
            "expected_cell_count": int(config["seed_count"]) * len(config["difficulty_separations"]),
            "completed_cell_count": completed_cell_count,
        },
        "cell_receipts": cell_receipts,
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
        "complete": True,
        "problems": [],
        "interpretation_limit": (
            "Generated latent data with provided diagnostic context labels only. This cannot establish "
            "learned routing, label-free deployment, natural-world generality, integrated cooperation, "
            "runtime activation, or substrate advantage."
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = _sealed(core, "result_sha256")
    atomic_write_json(out_path, result)
    return result


__all__ = [
    "ACTORS",
    "CELL_SCHEMA",
    "CLAIM_SCOPE",
    "CONFIG_SCHEMA",
    "CONTEXTS",
    "MAX_IDLE_WORKERS",
    "RESULT_SCHEMA",
    "SHARD_SCHEMA",
    "aggregate_result",
    "atomic_write_json",
    "canonical_sha256",
    "cell_path",
    "load_config",
    "run_cell",
    "run_shard",
    "validate_cell",
]
