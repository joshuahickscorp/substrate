
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.studies import generation1_context_routing as c2
from mop.substrate.events import canonical_bytes, canonical_sha256

CONFIG_SCHEMA = "mop-generation1-c3-dispatch-pilot-config/v1"
RESULT_SCHEMA = "mop-generation1-c3-dispatch-pilot/v1"
CLAIM_SCOPE = "generated-learned-dispatch-mechanics-pilot-only"

C2_RESULT_RELATIVE = Path("proof/GENERATION1_CONTEXT_ROUTING.json")
C2_VERIFICATION_RELATIVE = Path("proof/GENERATION1_CONTEXT_ROUTING.verification.json")
C2_RESULT_FILE_SHA256 = "4844ddffdf13db6bc8e5102e3095c4aa83ff5286bbf707e2b67d864c6db125f8"
C2_RESULT_SHA256 = "f7c8eae5a461a01b7489ecc99617918665a488ade90603c450e2c8fd5562dff7"
C2_VERIFICATION_FILE_SHA256 = "ac927e7fc39de83a48cfe33b90e493fd720a5ea86524226922c89e1dc32f5c52"
C2_VERIFICATION_SHA256 = "b653ffe87313b08ce31fe3ccbcb671e93fb2a2535c267b984306fd4e39a9caf2"

VISIBLE_ROUTER_INPUTS = ("latent_vector", "difficulty_index")
FORBIDDEN_HELDOUT_INPUTS = (
    "context_id",
    "truth",
    "actor_predictions",
    "oracle_actor_id",
    "heldout_route_labels",
)


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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_c2_authority(repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = repo_root / C2_RESULT_RELATIVE
    verification_path = repo_root / C2_VERIFICATION_RELATIVE
    if sha256_file(result_path) != C2_RESULT_FILE_SHA256:
        raise ValueError("C2 result file is not the exact frozen authority")
    if sha256_file(verification_path) != C2_VERIFICATION_FILE_SHA256:
        raise ValueError("C2 verification file is not the exact frozen authority")
    result = _read_object(result_path, "C2 result")
    verification = _read_object(verification_path, "C2 verification")
    _validate_seal(result, "result_sha256", "C2 result")
    _validate_seal(verification, "verification_sha256", "C2 verification")
    if result.get("result_sha256") != C2_RESULT_SHA256:
        raise ValueError("C2 result internal authority drifted")
    if verification.get("verification_sha256") != C2_VERIFICATION_SHA256:
        raise ValueError("C2 verification internal authority drifted")
    decision = result.get("decision") or {}
    mutation = verification.get("mutation_suite") or {}
    if (
        result.get("complete") is not True
        or result.get("problems") != []
        or decision.get("context_labeled_frozen_routing_confirmed") is not True
        or decision.get("ready_to_preregister_g1_c3_learned_dispatch") is not True
        or verification.get("verification_complete") is not True
        or verification.get("problems") != []
        or verification.get("result_sha256") != C2_RESULT_SHA256
        or mutation.get("all_rejected") is not True
    ):
        raise ValueError("C2 is not a clean independently verified prerequisite")
    binding = {
        "result_path": str(C2_RESULT_RELATIVE),
        "result_file_sha256": C2_RESULT_FILE_SHA256,
        "result_sha256": C2_RESULT_SHA256,
        "verification_path": str(C2_VERIFICATION_RELATIVE),
        "verification_file_sha256": C2_VERIFICATION_FILE_SHA256,
        "verification_sha256": C2_VERIFICATION_SHA256,
        "c2_ready_to_preregister": True,
        "c2_ready_to_train_confirmatory_dispatcher": bool(decision.get("ready_to_train_dispatcher")),
    }
    return binding, dict(result["config"])


def pilot_config(
    *,
    train_seed_start: int = 20270001,
    train_seed_count: int = 1,
    heldout_seed_start: int = 20271001,
    heldout_seed_count: int = 1,
    difficulty_indices: Sequence[int] = (0,),
    n_train: int = 120,
    n_test: int = 90,
    n_classes: int = 3,
    dim: int = 16,
    actor_epochs: int = 1,
    router_epochs: int = 12,
    router_hidden: int = 16,
) -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "campaign_id": "generation1-c3-d1-learned-dispatch-pilot-v1",
        "claim_scope": CLAIM_SCOPE,
        "train_seed_start": train_seed_start,
        "train_seed_count": train_seed_count,
        "heldout_seed_start": heldout_seed_start,
        "heldout_seed_count": heldout_seed_count,
        "difficulty_indices": [int(value) for value in difficulty_indices],
        "dataset": {
            "n_train": n_train,
            "n_test": n_test,
            "n_classes": n_classes,
            "dim": dim,
        },
        "actor_training": {"epochs": actor_epochs, "torch_threads": 1},
        "router": {
            "architecture": "mlp_one_hidden_layer",
            "hidden": router_hidden,
            "epochs": router_epochs,
            "learning_rate": 0.01,
            "training_seed": 31032026,
            "visible_inputs": list(VISIBLE_ROUTER_INPUTS),
            "forbidden_heldout_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        },
        "controls": [
            "global_static_actor",
            "per_difficulty_static_actor",
            "deterministic_random_actor",
            "fixed_c2_context_route_nonpromotable",
            "per_example_oracle_nonpromotable",
        ],
        "execution_class": "bounded_nonpromotable_mechanics_canary",
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def validate_config(config: Mapping[str, Any], c2_config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("C3 pilot schema or claim scope drifted")
    if config.get("activation_allowed") is not False or config.get("scientific_promotion") is not False:
        raise ValueError("C3 pilot activation or promotion escaped")
    for field in (
        "train_seed_start",
        "train_seed_count",
        "heldout_seed_start",
        "heldout_seed_count",
    ):
        if (
            isinstance(config.get(field), bool)
            or not isinstance(config.get(field), int)
            or config[field] <= 0
        ):
            raise ValueError(f"C3 {field} must be a positive integer")
    train = set(range(config["train_seed_start"], config["train_seed_start"] + config["train_seed_count"]))
    heldout = set(
        range(config["heldout_seed_start"], config["heldout_seed_start"] + config["heldout_seed_count"])
    )
    c2_seeds = set(
        range(int(c2_config["seed_start"]), int(c2_config["seed_start"]) + int(c2_config["seed_count"]))
    )
    c1_seeds = set(range(20260901, 20260949))
    if train & heldout or train & c2_seeds or heldout & c2_seeds or train & c1_seeds or heldout & c1_seeds:
        raise ValueError("C3 train/heldout seeds are not fresh and disjoint")
    difficulties = config.get("difficulty_indices")
    if (
        not isinstance(difficulties, list)
        or not difficulties
        or len(set(difficulties)) != len(difficulties)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= len(c2_config["difficulty_separations"])
            for value in difficulties
        )
    ):
        raise ValueError("C3 difficulty indexes are invalid")
    dataset = config.get("dataset") or {}
    if set(dataset) != {"n_train", "n_test", "n_classes", "dim"} or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dataset.values()
    ):
        raise ValueError("C3 pilot dataset is invalid")
    router = config.get("router") or {}
    if router.get("visible_inputs") != list(VISIBLE_ROUTER_INPUTS):
        raise ValueError("C3 router visible-input contract drifted")
    if router.get("forbidden_heldout_inputs") != list(FORBIDDEN_HELDOUT_INPUTS):
        raise ValueError("C3 heldout leakage guard drifted")
    if int(router.get("epochs", 0)) <= 0 or int(router.get("hidden", 0)) <= 0:
        raise ValueError("C3 router training is invalid")
    if (config.get("actor_training") or {}).get("torch_threads") != 1:
        raise ValueError("C3 actors require one thread each")


class _Dispatcher(torch.nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, len(c2.ACTORS))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _router_features(x: torch.Tensor, difficulty_index: int, difficulty_count: int) -> torch.Tensor:
    difficulty = float(difficulty_index) / max(1, difficulty_count - 1)
    column = torch.full((x.shape[0], 1), difficulty, dtype=x.dtype, device=x.device)
    return torch.cat((x, column), dim=1)


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_sha256(model: torch.nn.Module, mean: torch.Tensor, scale: torch.Tensor) -> str:
    tensors = [mean, scale]
    tensors.extend(value for _, value in sorted(model.state_dict().items()))
    return _tensor_sha256(*tensors)


def _fit_dispatcher(
    config: Mapping[str, Any], c2_config: Mapping[str, Any]
) -> tuple[_Dispatcher, torch.Tensor, torch.Tensor, dict[str, Any]]:
    dataset = config["dataset"]
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    difficulty_count = len(c2_config["difficulty_separations"])
    for seed in range(config["train_seed_start"], config["train_seed_start"] + config["train_seed_count"]):
        for difficulty_index in config["difficulty_indices"]:
            effective_seed = seed + difficulty_index * 1_000_003
            _, _, xte, _, contexts = c2.make_dataset(
                effective_seed,
                dataset["n_train"],
                dataset["n_test"],
                dataset["n_classes"],
                dataset["dim"],
                float(c2_config["difficulty_separations"][difficulty_index]),
            )
            features.append(_router_features(xte, difficulty_index, difficulty_count))
            actor_targets = [
                c2.ACTORS.index(c2_config["frozen_route"][c2.CONTEXTS[int(context)]][difficulty_index])
                for context in contexts.tolist()
            ]
            targets.append(torch.tensor(actor_targets, dtype=torch.long))
    x = torch.cat(features)
    y = torch.cat(targets)
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    router = config["router"]
    torch.manual_seed(int(router["training_seed"]))
    model = _Dispatcher(dataset["dim"], int(router["hidden"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(router["learning_rate"]))
    generator = torch.Generator().manual_seed(int(router["training_seed"]) + 1)
    for _ in range(int(router["epochs"])):
        for start in range(0, x.shape[0], 256):
            indexes = torch.randperm(x.shape[0], generator=generator)[start : start + 256]
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x[indexes]), y[indexes]).backward()
            optimizer.step()
    with torch.no_grad():
        train_accuracy = float((model(x).argmax(dim=1) == y).float().mean())
    audit = {
        "training_observations": int(x.shape[0]),
        "training_target": "c2_frozen_route_actor_from_training_context_only",
        "heldout_labels_used_for_training": False,
        "training_actor_target_accuracy": train_accuracy,
        "model_sha256": _model_sha256(model, mean, scale),
    }
    return model.eval(), mean, scale, audit


def _accuracy(prediction: Sequence[int], truth: Sequence[int]) -> float:
    return sum(int(left == right) for left, right in zip(prediction, truth, strict=True)) / len(truth)


def _evaluate_cell(
    config: Mapping[str, Any],
    c2_config: Mapping[str, Any],
    model: _Dispatcher,
    mean: torch.Tensor,
    scale: torch.Tensor,
    seed: int,
    difficulty_index: int,
) -> dict[str, Any]:
    dataset = config["dataset"]
    effective_seed = seed + difficulty_index * 1_000_003
    xtr, ytr, xte, yte, contexts = c2.make_dataset(
        effective_seed,
        dataset["n_train"],
        dataset["n_test"],
        dataset["n_classes"],
        dataset["dim"],
        float(c2_config["difficulty_separations"][difficulty_index]),
    )
    with torch.no_grad():
        router_input = _router_features(xte, difficulty_index, len(c2_config["difficulty_separations"]))
        selected_indexes = model((router_input - mean) / scale).argmax(dim=1).tolist()
    selected_actors = [c2.ACTORS[int(index)] for index in selected_indexes]
    predictions: dict[str, list[int]] = {}
    for actor_index, actor in enumerate(c2.ACTORS):
        prediction = c2.run_mode(
            actor,
            xtr,
            ytr,
            xte,
            dataset["n_classes"],
            int(config["actor_training"]["epochs"]),
            seed=effective_seed + (actor_index + 1) * 100_003,
        )
        predictions[actor] = [int(value) for value in prediction.tolist()]
    truth = [int(value) for value in yte.tolist()]
    context_ids = [int(value) for value in contexts.tolist()]
    learned = [predictions[selected_actors[index]][index] for index in range(len(truth))]
    global_actor = str(c2_config["controls"]["global_static_actor"])
    difficulty_actor = str(c2_config["controls"]["per_difficulty_static_actor"][difficulty_index])
    generator = torch.Generator().manual_seed(effective_seed + 11_000_019)
    random_indexes = torch.randint(0, len(c2.ACTORS), (len(truth),), generator=generator).tolist()
    random_prediction = [predictions[c2.ACTORS[int(index)]][row] for row, index in enumerate(random_indexes)]
    context_route = [
        predictions[c2_config["frozen_route"][c2.CONTEXTS[context]][difficulty_index]][row]
        for row, context in enumerate(context_ids)
    ]
    oracle = [
        truth[row]
        if any(predictions[actor][row] == truth[row] for actor in c2.ACTORS)
        else predictions[c2.ACTORS[0]][row]
        for row in range(len(truth))
    ]
    named = {
        "learned_dispatch": learned,
        "global_static": predictions[global_actor],
        "difficulty_static": predictions[difficulty_actor],
        "random_actor": random_prediction,
        "context_route_nonpromotable": context_route,
        "oracle_nonpromotable": oracle,
    }
    return {
        "seed": seed,
        "difficulty_index": difficulty_index,
        "effective_seed": effective_seed,
        "dataset_sha256": _tensor_sha256(xtr, ytr, xte, yte, contexts),
        "observation_count": len(truth),
        "accuracy": {name: _accuracy(values, truth) for name, values in named.items()},
        "selected_actor_counts": {actor: selected_actors.count(actor) for actor in c2.ACTORS},
        "router_received_fields": list(VISIBLE_ROUTER_INPUTS),
        "heldout_sensitive_payloads_emitted": False,
    }


def _mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def run_pilot(config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    torch.set_num_threads(1)
    binding, c2_config = load_c2_authority(repo_root)
    validate_config(config, c2_config)
    config_sha256 = canonical_sha256(config)
    model, mean, scale, training = _fit_dispatcher(config, c2_config)
    cells = [
        _evaluate_cell(config, c2_config, model, mean, scale, seed, difficulty)
        for seed in range(
            config["heldout_seed_start"], config["heldout_seed_start"] + config["heldout_seed_count"]
        )
        for difficulty in config["difficulty_indices"]
    ]
    arms = tuple(cells[0]["accuracy"])
    overall = {arm: _mean_ci([float(cell["accuracy"][arm]) for cell in cells]) for arm in arms}
    differences = {
        control: _mean_ci(
            [float(cell["accuracy"]["learned_dispatch"]) - float(cell["accuracy"][control]) for cell in cells]
        )
        for control in (
            "global_static",
            "difficulty_static",
            "random_actor",
            "context_route_nonpromotable",
        )
    }
    core = {
        "schema": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config": dict(config),
        "config_sha256": config_sha256,
        "c2_prerequisite": binding,
        "router_training": training,
        "heldout_contract": {
            "visible_inputs": list(VISIBLE_ROUTER_INPUTS),
            "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
            "contract_honored": True,
        },
        "grid": {
            "train_seed_count": config["train_seed_count"],
            "heldout_seed_count": config["heldout_seed_count"],
            "difficulty_count": len(config["difficulty_indices"]),
            "completed_cell_count": len(cells),
        },
        "cells": cells,
        "overall": overall,
        "learned_dispatch_differences": differences,
        "work_accounting": {
            "learned_dispatch_charged_actor_calls_per_observation": 1,
            "all_actor_control_calls_per_observation": len(c2.ACTORS),
            "control_generation_is_offline_experiment_cost": True,
        },
        "decision": {
            "mechanics_canary_complete": True,
            "advisory_learned_dispatch_beats_both_static_controls": (
                float(differences["global_static"]["mean"]) > 0
                and float(differences["difficulty_static"]["mean"]) > 0
            ),
            "ready_for_confirmatory_claim": False,
            "independent_verification_required": True,
        },
        "interpretation_limit": (
            "A downsized generated mechanics pilot can expose wiring or leakage failures only. "
            "It cannot confirm learned dispatch, generalization, activation, or substrate value."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "result_sha256")


def validate_result(
    result: Mapping[str, Any], config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    binding, c2_config = load_c2_authority(repo_root)
    validate_config(config, c2_config)
    _validate_seal(result, "result_sha256", "C3 pilot result")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("claim_scope") != CLAIM_SCOPE
        or result.get("config_sha256") != canonical_sha256(config)
        or result.get("c2_prerequisite") != binding
        or result.get("complete") is not True
        or result.get("problems") != []
        or result.get("activation_allowed") is not False
        or result.get("scientific_promotion") is not False
    ):
        raise ValueError("C3 pilot result identity, authority, or safety drifted")
    contract = result.get("heldout_contract") or {}
    if (
        contract.get("visible_inputs") != list(VISIBLE_ROUTER_INPUTS)
        or contract.get("forbidden_inputs") != list(FORBIDDEN_HELDOUT_INPUTS)
        or contract.get("contract_honored") is not True
    ):
        raise ValueError("C3 pilot heldout contract drifted")
    cells = result.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("C3 pilot has no heldout cells")
    for cell in cells:
        if cell.get("router_received_fields") != list(VISIBLE_ROUTER_INPUTS):
            raise ValueError("C3 pilot router input receipt drifted")
        if cell.get("heldout_sensitive_payloads_emitted") is not False:
            raise ValueError("C3 pilot emitted a heldout sensitive payload")
        if set(cell) & set(FORBIDDEN_HELDOUT_INPUTS):
            raise ValueError("C3 pilot cell contains forbidden heldout fields")
    decision = result.get("decision") or {}
    if (
        decision.get("ready_for_confirmatory_claim") is not False
        or decision.get("independent_verification_required") is not True
    ):
        raise ValueError("C3 pilot escaped its interpretation boundary")
