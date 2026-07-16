"""Exploratory G1-C3/D1 visible-router redesign screen.

This non-promotable screen compares several router feature/model variants while
sharing each heldout actor evaluation across the complete variant grid.  Every
router is restricted to the test latent, the frozen difficulty index, and
geometry derived from the labeled training support set.  Heldout context,
truth, actor predictions, and oracle choices are never router inputs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.studies import generation1_c3_dispatch as d1
from mop.studies import generation1_context_routing as c2

CONFIG_SCHEMA = "mop-generation1-c3-router-redesign-config/v1"
RESULT_SCHEMA = "mop-generation1-c3-router-redesign-rung/v1"
CLAIM_SCOPE = "generated-visible-router-redesign-exploration-only"

VISIBLE_ROUTER_INPUTS = (
    "latent_vector_when_variant_enabled",
    "difficulty_index",
    "labeled_training_support_geometry_when_variant_enabled",
)
FORBIDDEN_HELDOUT_INPUTS = d1.FORBIDDEN_HELDOUT_INPUTS
FEATURE_SETS = ("raw", "centroid", "local", "geometry", "raw_geometry")


def variant_grid() -> list[dict[str, Any]]:
    """Return the frozen exploratory grid in stable order."""

    return [
        {"variant_id": "raw-h32-e30-lr10", "feature_set": "raw", "hidden": 32, "epochs": 30, "lr": 0.01},
        {"variant_id": "raw-h64-e60-lr03", "feature_set": "raw", "hidden": 64, "epochs": 60, "lr": 0.003},
        {
            "variant_id": "centroid-h32-e30-lr10",
            "feature_set": "centroid",
            "hidden": 32,
            "epochs": 30,
            "lr": 0.01,
        },
        {
            "variant_id": "centroid-h64-e60-lr03",
            "feature_set": "centroid",
            "hidden": 64,
            "epochs": 60,
            "lr": 0.003,
        },
        {"variant_id": "local-h32-e30-lr10", "feature_set": "local", "hidden": 32, "epochs": 30, "lr": 0.01},
        {"variant_id": "local-h64-e60-lr03", "feature_set": "local", "hidden": 64, "epochs": 60, "lr": 0.003},
        {
            "variant_id": "geometry-h32-e30-lr10",
            "feature_set": "geometry",
            "hidden": 32,
            "epochs": 30,
            "lr": 0.01,
        },
        {
            "variant_id": "geometry-h64-e60-lr03",
            "feature_set": "geometry",
            "hidden": 64,
            "epochs": 60,
            "lr": 0.003,
        },
        {
            "variant_id": "raw-geometry-h32-e30-lr10",
            "feature_set": "raw_geometry",
            "hidden": 32,
            "epochs": 30,
            "lr": 0.01,
        },
        {
            "variant_id": "raw-geometry-h64-e60-lr03",
            "feature_set": "raw_geometry",
            "hidden": 64,
            "epochs": 60,
            "lr": 0.003,
        },
        {
            "variant_id": "geometry-h128-e90-lr03",
            "feature_set": "geometry",
            "hidden": 128,
            "epochs": 90,
            "lr": 0.003,
        },
        {
            "variant_id": "raw-geometry-h128-e90-lr03",
            "feature_set": "raw_geometry",
            "hidden": 128,
            "epochs": 90,
            "lr": 0.003,
        },
    ]


def redesign_config(
    *,
    train_seed_start: int = 20_300_001,
    train_seed_count: int = 3,
    heldout_seed_start: int = 20_310_001,
    heldout_seed_count: int = 3,
    difficulty_indices: Sequence[int] = (0, 1, 2, 3, 4),
    n_train: int = 360,
    n_test: int = 120,
    n_classes: int = 10,
    dim: int = 32,
    actor_epochs: int = 4,
    router_training_seed: int = 31_040_001,
    variants: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = [dict(value) for value in (variants if variants is not None else variant_grid())]
    return {
        "schema": CONFIG_SCHEMA,
        "campaign_id": "generation1-c3-d1-visible-router-redesign-v1",
        "claim_scope": CLAIM_SCOPE,
        "train_seed_start": train_seed_start,
        "train_seed_count": train_seed_count,
        "heldout_seed_start": heldout_seed_start,
        "heldout_seed_count": heldout_seed_count,
        "difficulty_indices": [int(value) for value in difficulty_indices],
        "dataset": {"n_train": n_train, "n_test": n_test, "n_classes": n_classes, "dim": dim},
        "actor_training": {"epochs": actor_epochs, "torch_threads": 1},
        "router_training_seed": router_training_seed,
        "variants": selected,
        "visible_inputs": list(VISIBLE_ROUTER_INPUTS),
        "forbidden_heldout_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "execution_class": "paired_nonpromotable_router_redesign_screen",
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def validate_config(config: Mapping[str, Any], c2_config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("router redesign schema or claim scope drifted")
    if config.get("activation_allowed") is not False or config.get("scientific_promotion") is not False:
        raise ValueError("router redesign activation or promotion escaped")
    for field in ("train_seed_start", "train_seed_count", "heldout_seed_start", "heldout_seed_count"):
        if (
            isinstance(config.get(field), bool)
            or not isinstance(config.get(field), int)
            or config[field] <= 0
        ):
            raise ValueError(f"router redesign {field} must be a positive integer")
    train = set(range(config["train_seed_start"], config["train_seed_start"] + config["train_seed_count"]))
    heldout = set(
        range(config["heldout_seed_start"], config["heldout_seed_start"] + config["heldout_seed_count"])
    )
    c2_seeds = set(
        range(int(c2_config["seed_start"]), int(c2_config["seed_start"]) + int(c2_config["seed_count"]))
    )
    prior_c3 = set(range(20_270_001, 20_280_001))
    if train & heldout or train & c2_seeds or heldout & c2_seeds or train & prior_c3 or heldout & prior_c3:
        raise ValueError("router redesign seeds are not fresh and disjoint")
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
        raise ValueError("router redesign difficulty indexes are invalid")
    dataset = config.get("dataset") or {}
    if set(dataset) != {"n_train", "n_test", "n_classes", "dim"} or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dataset.values()
    ):
        raise ValueError("router redesign dataset is invalid")
    if int(dataset["n_classes"]) < 2:
        raise ValueError("router redesign requires at least two classes")
    if (config.get("actor_training") or {}).get("torch_threads") != 1:
        raise ValueError("router redesign actors require one thread each")
    if (
        isinstance(config.get("router_training_seed"), bool)
        or not isinstance(config.get("router_training_seed"), int)
        or config["router_training_seed"] <= 0
    ):
        raise ValueError("router redesign training seed is invalid")
    if config.get("visible_inputs") != list(VISIBLE_ROUTER_INPUTS):
        raise ValueError("router redesign visible-input contract drifted")
    if config.get("forbidden_heldout_inputs") != list(FORBIDDEN_HELDOUT_INPUTS):
        raise ValueError("router redesign heldout leakage guard drifted")
    allowed = {row["variant_id"]: row for row in variant_grid()}
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("router redesign variant grid is empty")
    ids = [row.get("variant_id") if isinstance(row, dict) else None for row in variants]
    if len(ids) != len(set(ids)) or any(identifier not in allowed for identifier in ids):
        raise ValueError("router redesign variant inventory is invalid")
    if any(row != allowed[row["variant_id"]] for row in variants):
        raise ValueError("router redesign variant definition drifted")


def _centroids(xtr: torch.Tensor, ytr: torch.Tensor, n_classes: int) -> torch.Tensor:
    fallback = xtr.mean(dim=0)
    return torch.stack(
        [
            xtr[ytr == index].mean(dim=0) if bool((ytr == index).any()) else fallback
            for index in range(n_classes)
        ]
    )


def _scalar_geometry(x: torch.Tensor, difficulty_index: int, difficulty_count: int) -> torch.Tensor:
    difficulty = float(difficulty_index) / max(1, difficulty_count - 1)
    return torch.stack(
        (
            x.norm(dim=1),
            x.abs().mean(dim=1),
            x.std(dim=1, unbiased=False),
            torch.full((x.shape[0],), difficulty, dtype=x.dtype, device=x.device),
        ),
        dim=1,
    )


def _centroid_geometry(xtr: torch.Tensor, ytr: torch.Tensor, x: torch.Tensor, n_classes: int) -> torch.Tensor:
    centers = _centroids(xtr, ytr, n_classes)
    distances = torch.cdist(x, centers).sort(dim=1).values
    cosine = torch.nn.functional.cosine_similarity(x[:, None, :], centers[None, :, :], dim=2)
    absolute_cosine = cosine.abs().sort(dim=1, descending=True).values
    radii = []
    for index in range(n_classes):
        members = xtr[ytr == index]
        radii.append(
            (members - centers[index]).norm(dim=1).mean()
            if members.numel()
            else torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
    radius = torch.stack(radii).clamp_min(1e-6)
    normalized = (torch.cdist(x, centers) / radius[None, :]).sort(dim=1).values
    return torch.cat((distances, absolute_cosine, normalized), dim=1)


def _local_geometry(xtr: torch.Tensor, ytr: torch.Tensor, x: torch.Tensor, n_classes: int) -> torch.Tensor:
    distances = torch.cdist(x, xtr)
    k_max = min(15, xtr.shape[0])
    nearest_distance, nearest_index = distances.topk(k_max, largest=False)
    nearest_labels = ytr[nearest_index]
    columns: list[torch.Tensor] = []
    for requested in (1, 3, 5, 15):
        k = min(requested, k_max)
        columns.append(nearest_distance[:, :k].mean(dim=1))
    columns.append(nearest_distance.std(dim=1, unbiased=False))
    for requested in (3, 5, 15):
        k = min(requested, k_max)
        concentrations = torch.stack(
            [(nearest_labels[:, :k] == label).float().mean(dim=1) for label in range(n_classes)], dim=1
        )
        columns.append(concentrations.max(dim=1).values)
    probabilities = torch.stack(
        [(nearest_labels == label).float().mean(dim=1) for label in range(n_classes)], dim=1
    ).clamp_min(1e-12)
    columns.append(-(probabilities * probabilities.log()).sum(dim=1) / math.log(n_classes))
    return torch.stack(columns, dim=1)


def router_features(
    feature_set: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    x: torch.Tensor,
    *,
    difficulty_index: int,
    difficulty_count: int,
    n_classes: int,
) -> torch.Tensor:
    """Build the complete, label-safe heldout router interface."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown router feature set {feature_set!r}")
    scalar = _scalar_geometry(x, difficulty_index, difficulty_count)
    if feature_set == "raw":
        return torch.cat((x, scalar[:, -1:]), dim=1)
    centroid = None
    local = None
    if feature_set in {"centroid", "geometry", "raw_geometry"}:
        centroid = _centroid_geometry(xtr, ytr, x, n_classes)
    if feature_set in {"local", "geometry", "raw_geometry"}:
        local = _local_geometry(xtr, ytr, x, n_classes)
    if feature_set == "centroid":
        assert centroid is not None
        return torch.cat((scalar, centroid), dim=1)
    if feature_set == "local":
        assert local is not None
        return torch.cat((scalar, local), dim=1)
    assert centroid is not None and local is not None
    geometry = torch.cat((scalar, centroid, local), dim=1)
    return torch.cat((x, geometry), dim=1) if feature_set == "raw_geometry" else geometry


class _Router(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, len(c2.ACTORS)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_routers(
    config: Mapping[str, Any], c2_config: Mapping[str, Any]
) -> tuple[dict[str, tuple[_Router, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    dataset = config["dataset"]
    feature_sets = sorted({str(variant["feature_set"]) for variant in config["variants"]})
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in feature_sets}
    targets: list[torch.Tensor] = []
    difficulty_count = len(c2_config["difficulty_separations"])
    for seed in range(config["train_seed_start"], config["train_seed_start"] + config["train_seed_count"]):
        for difficulty_index in config["difficulty_indices"]:
            effective_seed = seed + difficulty_index * 1_000_003
            xtr, ytr, xte, _, contexts = c2.make_dataset(
                effective_seed,
                dataset["n_train"],
                dataset["n_test"],
                dataset["n_classes"],
                dataset["dim"],
                float(c2_config["difficulty_separations"][difficulty_index]),
            )
            for feature_set in feature_sets:
                chunks[feature_set].append(
                    router_features(
                        feature_set,
                        xtr,
                        ytr,
                        xte,
                        difficulty_index=difficulty_index,
                        difficulty_count=difficulty_count,
                        n_classes=dataset["n_classes"],
                    )
                )
            targets.append(
                torch.tensor(
                    [
                        c2.ACTORS.index(
                            c2_config["frozen_route"][c2.CONTEXTS[int(context)]][difficulty_index]
                        )
                        for context in contexts.tolist()
                    ],
                    dtype=torch.long,
                )
            )
    y = torch.cat(targets)
    feature_data: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for feature_set, values in chunks.items():
        x = torch.cat(values)
        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        feature_data[feature_set] = ((x - mean) / scale, mean, scale)
    fitted: dict[str, tuple[_Router, torch.Tensor, torch.Tensor]] = {}
    audits: dict[str, Any] = {}
    for variant_index, variant in enumerate(config["variants"]):
        variant_id = str(variant["variant_id"])
        x, mean, scale = feature_data[str(variant["feature_set"])]
        training_seed = int(config["router_training_seed"]) + variant_index * 100_003
        torch.manual_seed(training_seed)
        model = _Router(x.shape[1], int(variant["hidden"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(variant["lr"]))
        generator = torch.Generator().manual_seed(training_seed + 1)
        for _ in range(int(variant["epochs"])):
            permutation = torch.randperm(x.shape[0], generator=generator)
            for start in range(0, x.shape[0], 256):
                indexes = permutation[start : start + 256]
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(model(x[indexes]), y[indexes]).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            train_accuracy = float((model(x).argmax(dim=1) == y).float().mean())
        fitted[variant_id] = (model, mean, scale)
        audits[variant_id] = {
            "feature_set": variant["feature_set"],
            "feature_count": int(x.shape[1]),
            "training_observations": int(x.shape[0]),
            "training_target": "c2_frozen_route_actor_from_training_context_only",
            "heldout_labels_used_for_training": False,
            "training_actor_target_accuracy": train_accuracy,
            "model_sha256": d1._model_sha256(model, mean, scale),
        }
    return fitted, audits


def _evaluate_cell(
    config: Mapping[str, Any],
    c2_config: Mapping[str, Any],
    fitted: Mapping[str, tuple[_Router, torch.Tensor, torch.Tensor]],
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
    feature_cache = {
        feature_set: router_features(
            feature_set,
            xtr,
            ytr,
            xte,
            difficulty_index=difficulty_index,
            difficulty_count=len(c2_config["difficulty_separations"]),
            n_classes=dataset["n_classes"],
        )
        for feature_set in {str(variant["feature_set"]) for variant in config["variants"]}
    }
    selected: dict[str, list[str]] = {}
    with torch.no_grad():
        for variant in config["variants"]:
            variant_id = str(variant["variant_id"])
            model, mean, scale = fitted[variant_id]
            indexes = model((feature_cache[str(variant["feature_set"])] - mean) / scale).argmax(dim=1)
            selected[variant_id] = [c2.ACTORS[int(index)] for index in indexes.tolist()]
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
    variant_accuracy = {
        variant_id: d1._accuracy([predictions[actors[row]][row] for row in range(len(truth))], truth)
        for variant_id, actors in selected.items()
    }
    global_actor = str(c2_config["controls"]["global_static_actor"])
    difficulty_actor = str(c2_config["controls"]["per_difficulty_static_actor"][difficulty_index])
    generator = torch.Generator().manual_seed(effective_seed + 11_000_019)
    random_indexes = torch.randint(0, len(c2.ACTORS), (len(truth),), generator=generator).tolist()
    control_predictions = {
        "global_static": predictions[global_actor],
        "difficulty_static": predictions[difficulty_actor],
        "random_actor": [predictions[c2.ACTORS[int(index)]][row] for row, index in enumerate(random_indexes)],
        "context_route_nonpromotable": [
            predictions[c2_config["frozen_route"][c2.CONTEXTS[context]][difficulty_index]][row]
            for row, context in enumerate(context_ids)
        ],
        "oracle_nonpromotable": [
            truth[row]
            if any(predictions[actor][row] == truth[row] for actor in c2.ACTORS)
            else predictions[c2.ACTORS[0]][row]
            for row in range(len(truth))
        ],
    }
    return {
        "seed": seed,
        "difficulty_index": difficulty_index,
        "effective_seed": effective_seed,
        "dataset_sha256": d1._tensor_sha256(xtr, ytr, xte, yte, contexts),
        "observation_count": len(truth),
        "variant_accuracy": variant_accuracy,
        "control_accuracy": {
            name: d1._accuracy(values, truth) for name, values in control_predictions.items()
        },
        "selected_actor_counts": {
            variant_id: {actor: actors.count(actor) for actor in c2.ACTORS}
            for variant_id, actors in selected.items()
        },
        "router_received_fields": list(VISIBLE_ROUTER_INPUTS),
        "heldout_sensitive_payloads_emitted": False,
    }


def _mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def summarize_cells(cells: Sequence[Mapping[str, Any]], variant_ids: Sequence[str]) -> dict[str, Any]:
    controls = tuple(cells[0]["control_accuracy"])
    control_summary = {
        name: _mean_ci([float(cell["control_accuracy"][name]) for cell in cells]) for name in controls
    }
    variants = {}
    for variant_id in variant_ids:
        accuracy = _mean_ci([float(cell["variant_accuracy"][variant_id]) for cell in cells])
        differences = {
            control: _mean_ci(
                [
                    float(cell["variant_accuracy"][variant_id]) - float(cell["control_accuracy"][control])
                    for cell in cells
                ]
            )
            for control in (
                "global_static",
                "difficulty_static",
                "random_actor",
                "context_route_nonpromotable",
            )
        }
        variants[variant_id] = {"learned_dispatch": accuracy, "differences": differences}
    ranking = sorted(
        variant_ids,
        key=lambda identifier: (
            min(
                float(variants[identifier]["differences"]["global_static"]["mean"]),
                float(variants[identifier]["differences"]["difficulty_static"]["mean"]),
            ),
            float(variants[identifier]["learned_dispatch"]["mean"]),
            identifier,
        ),
        reverse=True,
    )
    return {"controls": control_summary, "variants": variants, "ranking": ranking}


def run_redesign(config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    torch.set_num_threads(1)
    binding, c2_config = d1.load_c2_authority(repo_root)
    validate_config(config, c2_config)
    fitted, audits = _fit_routers(config, c2_config)
    cells = [
        _evaluate_cell(config, c2_config, fitted, seed, difficulty)
        for seed in range(
            config["heldout_seed_start"], config["heldout_seed_start"] + config["heldout_seed_count"]
        )
        for difficulty in config["difficulty_indices"]
    ]
    variant_ids = [str(variant["variant_id"]) for variant in config["variants"]]
    summary = summarize_cells(cells, variant_ids)
    best = summary["ranking"][0]
    core = {
        "schema": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config": dict(config),
        "config_sha256": d1.canonical_sha256(config),
        "c2_prerequisite": binding,
        "router_training": audits,
        "heldout_contract": {
            "visible_inputs": list(VISIBLE_ROUTER_INPUTS),
            "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
            "contract_honored": True,
        },
        "grid": {
            "variant_count": len(variant_ids),
            "train_seed_count": config["train_seed_count"],
            "heldout_seed_count": config["heldout_seed_count"],
            "difficulty_count": len(config["difficulty_indices"]),
            "completed_cell_count": len(cells),
            "shared_actor_evaluation": True,
        },
        "cells": cells,
        "summary": summary,
        "decision": {
            "best_exploratory_variant": best,
            "redesign_screen_rung_complete": True,
            "ready_for_confirmatory_claim": False,
            "independent_verification_required": True,
        },
        "interpretation_limit": (
            "This paired generated redesign screen is tuning evidence only. It cannot confirm learned "
            "dispatch, authorize activation, or promote substrate science."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return d1._sealed(core, "result_sha256")


def validate_result(
    result: Mapping[str, Any], config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    binding, c2_config = d1.load_c2_authority(repo_root)
    validate_config(config, c2_config)
    d1._validate_seal(result, "result_sha256", "router redesign result")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("claim_scope") != CLAIM_SCOPE
        or result.get("config_sha256") != d1.canonical_sha256(config)
        or result.get("c2_prerequisite") != binding
        or result.get("complete") is not True
        or result.get("problems") != []
        or result.get("activation_allowed") is not False
        or result.get("scientific_promotion") is not False
    ):
        raise ValueError("router redesign result identity, authority, or safety drifted")
    contract = result.get("heldout_contract") or {}
    if (
        contract.get("visible_inputs") != list(VISIBLE_ROUTER_INPUTS)
        or contract.get("forbidden_inputs") != list(FORBIDDEN_HELDOUT_INPUTS)
        or contract.get("contract_honored") is not True
    ):
        raise ValueError("router redesign heldout contract drifted")
    cells = result.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("router redesign has no heldout cells")
    expected_ids = {str(variant["variant_id"]) for variant in config["variants"]}
    for cell in cells:
        if set(cell.get("variant_accuracy") or {}) != expected_ids:
            raise ValueError("router redesign variant receipt drifted")
        if cell.get("router_received_fields") != list(VISIBLE_ROUTER_INPUTS):
            raise ValueError("router redesign input receipt drifted")
        if cell.get("heldout_sensitive_payloads_emitted") is not False:
            raise ValueError("router redesign emitted a heldout sensitive payload")
        if set(cell) & set(FORBIDDEN_HELDOUT_INPUTS):
            raise ValueError("router redesign cell contains forbidden heldout fields")
    decision = result.get("decision") or {}
    if (
        decision.get("ready_for_confirmatory_claim") is not False
        or decision.get("independent_verification_required") is not True
    ):
        raise ValueError("router redesign escaped its interpretation boundary")
