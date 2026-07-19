"""Generation 1 C1 difficulty and actor-competence atlas.

This study expands the earlier PR1 generated task into a sealed, restart-safe
task-family x context x difficulty x actor tensor.  It is deliberately scoped
to generated latent classification.  A positive result may license a C2
complementarity preregistration, but never runtime activation or a substrate
claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.substrate.events import canonical_bytes, canonical_sha256

_pr1: Any = import_module("scripts.pr1_mode_error_disjointness")
MODES = tuple(str(value) for value in _pr1.MODES)
SUBPOPS = tuple(str(value) for value in _pr1.SUBPOPS)
make_dataset = _pr1.make_dataset
predict_prototype = _pr1.predict_prototype
run_mode = _pr1.run_mode

CONFIG_SCHEMA = "mop-generation1-competence-atlas-config/v1"
SEED_SCHEMA = "mop-generation1-competence-atlas-seed/v1"
ATLAS_SCHEMA = "mop-generation1-competence-atlas/v1"
CLAIM_SCOPE = "generated-latent-actor-context-niches-only"
TASK_FAMILY = "generated-mixed-multiclass-latents"
ACTORS = tuple(MODES)
CONTEXTS = tuple(SUBPOPS)
MAX_SEED_WORKERS = 6






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
    observed = value.get(field)
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if observed != expected:
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


def load_config(path: Path | str) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    config = _read_json(source, "competence-atlas config")
    _exact_keys(
        config,
        {
            "schema",
            "campaign_id",
            "claim_scope",
            "prerequisite",
            "seeds",
            "difficulty_separations",
            "dataset",
            "training",
            "criteria",
            "controls",
            "activation_allowed",
            "scientific_promotion",
        },
        "competence-atlas config",
    )
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError("unexpected competence-atlas config schema")
    if config["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("competence-atlas claim scope drifted")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 8
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("competence atlas requires at least eight distinct integer seeds")
    separations = config["difficulty_separations"]
    if (
        not isinstance(separations, list)
        or len(separations) < 3
        or any(not isinstance(value, int | float) or float(value) <= 0 for value in separations)
        or len(separations) != len(set(float(value) for value in separations))
    ):
        raise ValueError("competence atlas requires at least three positive difficulty values")
    dataset = config["dataset"]
    _exact_keys(dataset, {"n_train", "n_test", "n_classes", "dim"}, "dataset config")
    if any(int(dataset[key]) <= 0 for key in dataset):
        raise ValueError("dataset dimensions must be positive")
    if int(dataset["n_classes"]) < 2 or int(dataset["n_test"]) < 3:
        raise ValueError("dataset class or test count is too small")
    training = config["training"]
    _exact_keys(
        training,
        {"epochs", "homogeneous_actor", "homogeneous_copies", "torch_threads"},
        "training config",
    )
    if int(training["epochs"]) <= 0 or int(training["homogeneous_copies"]) < 2:
        raise ValueError("training epochs and homogeneous-copy count must be positive")
    if training["homogeneous_actor"] not in ACTORS:
        raise ValueError("unknown homogeneous control actor")
    if int(training["torch_threads"]) != 1:
        raise ValueError("official competence atlas is restricted to one torch thread")
    criteria = config["criteria"]
    _exact_keys(
        criteria,
        {
            "min_niche_advantage",
            "min_oracle_headroom",
            "min_reproducible_fraction",
            "off_ceiling_max_accuracy",
            "above_chance_margin",
        },
        "criteria config",
    )
    for key in criteria:
        value = float(criteria[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"criteria {key} must be in [0, 1]")
    controls = config["controls"]
    _exact_keys(
        controls,
        {"best_single", "random", "homogeneous", "oracle_actor", "abstention"},
        "control declaration",
    )
    if not all(controls.values()):
        raise ValueError("all G1-C1 controls must remain enabled")
    if config["activation_allowed"] is not False or config["scientific_promotion"] is not False:
        raise ValueError("competence-atlas activation or promotion escaped")
    prerequisite = config["prerequisite"]
    _exact_keys(
        prerequisite,
        {
            "synthesis_path",
            "synthesis_file_sha256",
            "synthesis_sha256",
            "verification_path",
            "verification_file_sha256",
            "verification_sha256",
        },
        "prerequisite binding",
    )
    return config, sha256_file(source)


def validate_prerequisite(config: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    binding = config["prerequisite"]
    synthesis_path = (repo_root / str(binding["synthesis_path"])).resolve()
    verification_path = (repo_root / str(binding["verification_path"])).resolve()
    for path in (synthesis_path, verification_path):
        if not path.is_relative_to(repo_root.resolve()):
            raise ValueError("prerequisite path escapes the repository")
    synthesis = _read_json(synthesis_path, "G1-C0 synthesis")
    verification = _read_json(verification_path, "G1-C0 verification")
    if sha256_file(synthesis_path) != binding["synthesis_file_sha256"]:
        raise ValueError("G1-C0 synthesis file authority drifted")
    if sha256_file(verification_path) != binding["verification_file_sha256"]:
        raise ValueError("G1-C0 verification file authority drifted")
    if synthesis.get("synthesis_sha256") != binding["synthesis_sha256"]:
        raise ValueError("G1-C0 synthesis payload authority drifted")
    if verification.get("verification_sha256") != binding["verification_sha256"]:
        raise ValueError("G1-C0 verification payload authority drifted")
    if verification.get("verification_complete") is not True or verification.get("problems") != []:
        raise ValueError("G1-C0 independent verification is not clean")
    if any(
        row.get("activation_allowed") is not False or row.get("scientific_promotion") is not False
        for row in (synthesis, verification)
    ):
        raise ValueError("G1-C0 claim boundary drifted")
    boundary = (synthesis.get("claim_boundaries") or {}).get("context_disjoint_actor_niches") or {}
    if boundary.get("status") != "not_tested_by_g1_c0":
        raise ValueError("G1-C0 no longer licenses the bounded C1 question")
    return {
        "synthesis_path": str(binding["synthesis_path"]),
        "synthesis_file_sha256": str(binding["synthesis_file_sha256"]),
        "synthesis_sha256": str(binding["synthesis_sha256"]),
        "verification_path": str(binding["verification_path"]),
        "verification_file_sha256": str(binding["verification_file_sha256"]),
        "verification_sha256": str(binding["verification_sha256"]),
    }


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


def _accuracy(prediction: Sequence[int], truth: Sequence[int], mask: Sequence[bool]) -> float:
    selected = [index for index, include in enumerate(mask) if include]
    if not selected:
        return 0.0
    return sum(int(prediction[index] == truth[index]) for index in selected) / len(selected)


def _majority_abstention(
    predictions: Mapping[str, list[int]],
    truth: list[int],
    mask: list[bool],
) -> dict[str, Any]:
    accepted = 0
    correct = 0
    for index, include in enumerate(mask):
        if not include:
            continue
        counts: dict[int, int] = {}
        for actor in ACTORS:
            label = predictions[actor][index]
            counts[label] = counts.get(label, 0) + 1
        label, votes = max(counts.items(), key=lambda item: (item[1], -item[0]))
        if votes >= 3:
            accepted += 1
            correct += int(label == truth[index])
    total = sum(mask)
    return {
        "coverage": 0.0 if total == 0 else accepted / total,
        "selective_accuracy": None if accepted == 0 else correct / accepted,
        "accepted": accepted,
        "total": total,
        "vote_threshold": 3,
    }


def _context_metrics(
    truth: list[int],
    context_ids: list[int],
    predictions: Mapping[str, list[int]],
    homogeneous_predictions: Sequence[list[int]],
    random_prediction: list[int],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for context_index, context in enumerate(CONTEXTS):
        mask = [value == context_index for value in context_ids]
        actor_accuracy = {actor: _accuracy(predictions[actor], truth, mask) for actor in ACTORS}
        oracle_prediction = [
            truth[index]
            if any(predictions[actor][index] == truth[index] for actor in ACTORS)
            else predictions[ACTORS[0]][index]
            for index in range(len(truth))
        ]
        homogeneous_oracle = [
            truth[index]
            if any(copy[index] == truth[index] for copy in homogeneous_predictions)
            else homogeneous_predictions[0][index]
            for index in range(len(truth))
        ]
        rows[context] = {
            "observation_count": sum(mask),
            "actor_accuracy": actor_accuracy,
            "best_single_accuracy": max(actor_accuracy.values()),
            "random_accuracy": _accuracy(random_prediction, truth, mask),
            "oracle_actor_accuracy": _accuracy(oracle_prediction, truth, mask),
            "homogeneous_copy_mean_accuracy": statistics.fmean(
                _accuracy(copy, truth, mask) for copy in homogeneous_predictions
            ),
            "homogeneous_oracle_accuracy": _accuracy(homogeneous_oracle, truth, mask),
            "abstention": _majority_abstention(predictions, truth, mask),
        }
    return rows


def run_seed(config: Mapping[str, Any], config_sha256: str, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    dataset = config["dataset"]
    training = config["training"]
    difficulties: list[dict[str, Any]] = []
    seed_started = time.perf_counter()
    for difficulty_index, separation_value in enumerate(config["difficulty_separations"]):
        separation = float(separation_value)
        effective_seed = seed + difficulty_index * 1_000_003
        xtr, ytr, xte, yte, context_ids_tensor = make_dataset(
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
            started = time.perf_counter()
            prediction = run_mode(
                actor,
                xtr,
                ytr,
                xte,
                int(dataset["n_classes"]),
                int(training["epochs"]),
                seed=effective_seed + (actor_index + 1) * 100_003,
            )
            actor_seconds[actor] = time.perf_counter() - started
            predictions[actor] = [int(value) for value in prediction.tolist()]
        homogeneous_predictions: list[list[int]] = []
        for copy_index in range(int(training["homogeneous_copies"])):
            prediction = run_mode(
                str(training["homogeneous_actor"]),
                xtr,
                ytr,
                xte,
                int(dataset["n_classes"]),
                int(training["epochs"]),
                seed=effective_seed + 5_000_021 + copy_index * 200_003,
            )
            homogeneous_predictions.append([int(value) for value in prediction.tolist()])
        generator = torch.Generator().manual_seed(effective_seed + 9_000_011)
        random_prediction = [
            int(value)
            for value in torch.randint(
                0,
                int(dataset["n_classes"]),
                (len(yte),),
                generator=generator,
            ).tolist()
        ]
        truth = [int(value) for value in yte.tolist()]
        context_ids = [int(value) for value in context_ids_tensor.tolist()]
        difficulties.append(
            {
                "difficulty_index": difficulty_index,
                "separation": separation,
                "effective_seed": effective_seed,
                "dataset_sha256": _tensor_sha256(xtr, ytr, xte, yte, context_ids_tensor),
                "truth": truth,
                "context_ids": context_ids,
                "predictions": predictions,
                "homogeneous_predictions": homogeneous_predictions,
                "random_prediction": random_prediction,
                "context_metrics": _context_metrics(
                    truth,
                    context_ids,
                    predictions,
                    homogeneous_predictions,
                    random_prediction,
                ),
                "actor_wall_seconds": actor_seconds,
                "peak_process_rss_bytes": _peak_rss_bytes(),
            }
        )
    core = {
        "schema": SEED_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_file_sha256": config_sha256,
        "seed": seed,
        "difficulty_count": len(difficulties),
        "difficulties": difficulties,
        "wall_seconds": time.perf_counter() - seed_started,
        "complete": True,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "seed_sha256")


def validate_seed_receipt(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    expected_seed: int,
) -> None:
    if receipt.get("schema") != SEED_SCHEMA:
        raise ValueError("seed receipt schema drifted")
    _validate_seal(receipt, "seed_sha256", "seed receipt")
    if receipt.get("campaign_id") != config["campaign_id"]:
        raise ValueError("seed receipt campaign drifted")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("seed receipt claim scope drifted")
    if receipt.get("config_file_sha256") != config_sha256:
        raise ValueError("seed receipt config authority drifted")
    if receipt.get("seed") != expected_seed or receipt.get("complete") is not True:
        raise ValueError("seed receipt identity or completion drifted")
    if receipt.get("activation_allowed") is not False or receipt.get("scientific_promotion") is not False:
        raise ValueError("seed receipt activation or promotion escaped")
    rows = receipt.get("difficulties")
    if not isinstance(rows, list) or len(rows) != len(config["difficulty_separations"]):
        raise ValueError("seed receipt difficulty inventory drifted")
    n_test = int(config["dataset"]["n_test"])
    copies = int(config["training"]["homogeneous_copies"])
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("seed difficulty row is invalid")
        if row.get("difficulty_index") != index or not math.isclose(
            float(row.get("separation", -1)),
            float(config["difficulty_separations"][index]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("seed difficulty coordinate drifted")
        truth = row.get("truth")
        context_ids = row.get("context_ids")
        predictions = row.get("predictions")
        homogeneous = row.get("homogeneous_predictions")
        random_prediction = row.get("random_prediction")
        if not isinstance(truth, list) or len(truth) != n_test:
            raise ValueError("seed truth vector length drifted")
        if not isinstance(context_ids, list) or len(context_ids) != n_test:
            raise ValueError("seed context vector length drifted")
        if set(context_ids) != set(range(len(CONTEXTS))):
            raise ValueError("seed context inventory drifted")
        if not isinstance(predictions, dict) or set(predictions) != set(ACTORS):
            raise ValueError("seed actor inventory drifted")
        if any(
            not isinstance(predictions[actor], list) or len(predictions[actor]) != n_test for actor in ACTORS
        ):
            raise ValueError("seed actor prediction length drifted")
        if (
            not isinstance(homogeneous, list)
            or len(homogeneous) != copies
            or any(not isinstance(copy, list) or len(copy) != n_test for copy in homogeneous)
        ):
            raise ValueError("seed homogeneous control drifted")
        if not isinstance(random_prediction, list) or len(random_prediction) != n_test:
            raise ValueError("seed random control drifted")
        expected_metrics = _context_metrics(
            truth,
            context_ids,
            predictions,
            homogeneous,
            random_prediction,
        )
        if row.get("context_metrics") != expected_metrics:
            raise ValueError("seed context metrics do not rebuild from raw predictions")


def _interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def aggregate_receipts(
    config: Mapping[str, Any],
    config_sha256: str,
    prerequisite: Mapping[str, Any],
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    complete: bool,
) -> dict[str, Any]:
    criteria = config["criteria"]
    chance = 1.0 / int(config["dataset"]["n_classes"])
    tensor: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    niche_candidates: list[dict[str, Any]] = []
    headroom_candidates: list[dict[str, Any]] = []
    for difficulty_index, separation_value in enumerate(config["difficulty_separations"]):
        for context in CONTEXTS:
            per_seed_accuracy: dict[str, list[float]] = {actor: [] for actor in ACTORS}
            per_seed_best: list[float] = []
            per_seed_oracle: list[float] = []
            per_seed_random: list[float] = []
            per_seed_homogeneous: list[float] = []
            per_seed_abstention_coverage: list[float] = []
            per_seed_abstention_accuracy: list[float] = []
            for _, receipt in receipts:
                metrics = receipt["difficulties"][difficulty_index]["context_metrics"][context]
                actor_accuracy = metrics["actor_accuracy"]
                for actor in ACTORS:
                    per_seed_accuracy[actor].append(float(actor_accuracy[actor]))
                per_seed_best.append(max(float(value) for value in actor_accuracy.values()))
                per_seed_oracle.append(float(metrics["oracle_actor_accuracy"]))
                per_seed_random.append(float(metrics["random_accuracy"]))
                per_seed_homogeneous.append(float(metrics["homogeneous_oracle_accuracy"]))
                abstention = metrics["abstention"]
                per_seed_abstention_coverage.append(float(abstention["coverage"]))
                if abstention["selective_accuracy"] is not None:
                    per_seed_abstention_accuracy.append(float(abstention["selective_accuracy"]))
            winner_mean = (
                max((statistics.fmean(values), actor) for actor, values in per_seed_accuracy.items())[1]
                if receipts
                else None
            )
            for actor in ACTORS:
                accuracies = per_seed_accuracy[actor]
                deltas = [
                    accuracy - max(per_seed_accuracy[other][seed_index] for other in ACTORS if other != actor)
                    for seed_index, accuracy in enumerate(accuracies)
                ]
                quality = _interval(accuracies)
                advantage = _interval(deltas)
                off_ceiling = bool(
                    quality["mean"] is not None
                    and chance + float(criteria["above_chance_margin"])
                    <= float(quality["mean"])
                    <= float(criteria["off_ceiling_max_accuracy"])
                )
                favorable_fraction = 0.0 if not deltas else sum(delta > 0 for delta in deltas) / len(deltas)
                candidate = bool(
                    actor == winner_mean
                    and off_ceiling
                    and advantage["lo"] is not None
                    and float(advantage["lo"]) > 0
                    and float(advantage["mean"]) >= float(criteria["min_niche_advantage"])
                    and favorable_fraction >= float(criteria["min_reproducible_fraction"])
                )
                row = {
                    "task_family": TASK_FAMILY,
                    "context": context,
                    "difficulty_index": difficulty_index,
                    "separation": float(separation_value),
                    "actor": actor,
                    "quality": quality,
                    "advantage_over_next_best": advantage,
                    "favorable_seed_fraction": favorable_fraction,
                    "off_ceiling": off_ceiling,
                    "niche_candidate": candidate,
                }
                tensor.append(row)
                if candidate:
                    niche_candidates.append(row)
            headroom_values = [
                oracle - best for oracle, best in zip(per_seed_oracle, per_seed_best, strict=True)
            ]
            headroom = _interval(headroom_values)
            headroom_candidate = bool(
                headroom["lo"] is not None
                and float(headroom["lo"]) > 0
                and float(headroom["mean"]) >= float(criteria["min_oracle_headroom"])
            )
            control_row = {
                "task_family": TASK_FAMILY,
                "context": context,
                "difficulty_index": difficulty_index,
                "separation": float(separation_value),
                "best_single": _interval(per_seed_best),
                "random": _interval(per_seed_random),
                "homogeneous_oracle": _interval(per_seed_homogeneous),
                "oracle_actor": _interval(per_seed_oracle),
                "oracle_headroom_over_best_single": headroom,
                "oracle_headroom_candidate": headroom_candidate,
                "abstention_coverage": _interval(per_seed_abstention_coverage),
                "abstention_selective_accuracy": _interval(per_seed_abstention_accuracy),
            }
            controls.append(control_row)
            if headroom_candidate:
                headroom_candidates.append(control_row)
    actor_contexts: dict[str, set[str]] = {}
    for row in niche_candidates:
        actor_contexts.setdefault(str(row["actor"]), set()).add(str(row["context"]))
    disjoint_pairs = []
    actors = sorted(actor_contexts)
    for index, left in enumerate(actors):
        for right in actors[index + 1 :]:
            left_only = sorted(actor_contexts[left] - actor_contexts[right])
            right_only = sorted(actor_contexts[right] - actor_contexts[left])
            if left_only and right_only:
                disjoint_pairs.append(
                    {
                        "left_actor": left,
                        "right_actor": right,
                        "left_only_contexts": left_only,
                        "right_only_contexts": right_only,
                    }
                )
    expected_receipts = len(config["seeds"])
    expected_cells = expected_receipts * len(config["difficulty_separations"])
    bed_valid = bool(niche_candidates and headroom_candidates)
    c2_ready = bool(complete and bed_valid and disjoint_pairs)
    decision = {
        "bed_valid": bed_valid,
        "off_ceiling_cell_count": sum(row["off_ceiling"] for row in tensor),
        "niche_candidate_count": len(niche_candidates),
        "niche_actor_ids": sorted(actor_contexts),
        "oracle_headroom_cell_count": len(headroom_candidates),
        "disjoint_actor_pairs": disjoint_pairs,
        "reproducible_context_disjoint_niches": bool(disjoint_pairs),
        "ready_to_preregister_g1_c2": c2_ready,
        "ready_to_train_dispatcher": False,
        "verdict": (
            "generated_bed_disjoint_niches_candidate_pending_independent_verification"
            if c2_ready
            else ("generated_bed_complete_without_c2_license" if complete else "generated_bed_in_progress")
        ),
    }
    core = {
        "schema": ATLAS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "task_family": TASK_FAMILY,
        "prerequisite": dict(prerequisite),
        "config": {
            "file_sha256": config_sha256,
            "seeds": list(config["seeds"]),
            "difficulty_separations": [float(value) for value in config["difficulty_separations"]],
            "actors": list(ACTORS),
            "contexts": list(CONTEXTS),
            "dataset": dict(config["dataset"]),
            "training": dict(config["training"]),
            "criteria": dict(config["criteria"]),
            "controls": dict(config["controls"]),
        },
        "grid": {
            "expected_seed_count": expected_receipts,
            "completed_seed_count": len(receipts),
            "expected_seed_difficulty_cells": expected_cells,
            "completed_seed_difficulty_cells": len(receipts) * len(config["difficulty_separations"]),
        },
        "seed_receipts": [
            {
                "path": path,
                "seed": receipt["seed"],
                "seed_sha256": receipt["seed_sha256"],
            }
            for path, receipt in receipts
        ],
        "competence_tensor": tensor,
        "controls": controls,
        "decision": decision,
        "complete": complete,
        "problems": [],
        "interpretation_limit": (
            "Generated latent actors and contexts only. This atlas does not establish natural-world "
            "generality, integrated cooperation, learned routing, or substrate advantage."
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "atlas_sha256")


def run_atlas(
    config_path: Path | str,
    work_root: Path | str,
    out_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
    max_new_seeds: int | None = None,
    seed_workers: int = 1,
) -> dict[str, Any]:
    if isinstance(seed_workers, bool) or not isinstance(seed_workers, int):
        raise ValueError("seed workers must be an integer")
    if not 1 <= seed_workers <= MAX_SEED_WORKERS:
        raise ValueError(f"seed workers must be in [1, {MAX_SEED_WORKERS}]")
    if max_new_seeds is not None and (
        isinstance(max_new_seeds, bool) or not isinstance(max_new_seeds, int) or max_new_seeds < 0
    ):
        raise ValueError("max new seeds must be a nonnegative integer or null")
    config, config_sha256 = load_config(config_path)
    prerequisite = validate_prerequisite(config, repo_root=repo_root)
    root = Path(work_root)
    receipts_dir = root / "seeds"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_by_seed: dict[int, dict[str, Any]] = {}
    path_by_seed: dict[int, Path] = {}
    for seed in config["seeds"]:
        path = receipts_dir / f"seed_{seed}.json"
        path_by_seed[seed] = path
        if path.is_file():
            try:
                candidate = _read_json(path, f"seed {seed} receipt")
                validate_seed_receipt(candidate, config, config_sha256, seed)
                receipt_by_seed[seed] = candidate
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    def ordered_receipts() -> list[tuple[str, Mapping[str, Any]]]:
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for seed in config["seeds"]:
            receipt = receipt_by_seed.get(seed)
            if receipt is None:
                continue
            path = path_by_seed[seed].resolve()
            relative = (
                str(path.relative_to(repo_root.resolve()))
                if path.is_relative_to(repo_root.resolve())
                else str(path)
            )
            rows.append((relative, receipt))
        return rows

    def publish(seed: int) -> None:
        receipts = ordered_receipts()
        partial = aggregate_receipts(
            config,
            config_sha256,
            prerequisite,
            receipts,
            complete=len(receipts) == len(config["seeds"]),
        )
        atomic_write_json(out_path, partial)
        print(
            f"G1-C1 seed {seed} complete: {len(receipts)}/{len(config['seeds'])}",
            flush=True,
        )

    missing = [seed for seed in config["seeds"] if seed not in receipt_by_seed]
    scheduled = missing if max_new_seeds is None else missing[:max_new_seeds]
    if seed_workers == 1:
        for seed in scheduled:
            receipt = run_seed(config, config_sha256, seed)
            validate_seed_receipt(receipt, config, config_sha256, seed)
            atomic_write_json(path_by_seed[seed], receipt)
            receipt_by_seed[seed] = receipt
            publish(seed)
    elif scheduled:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=seed_workers, mp_context=context) as executor:
            futures = {
                executor.submit(run_seed, config, config_sha256, seed): seed for seed in scheduled
            }
            for future in as_completed(futures):
                seed = futures[future]
                receipt = future.result()
                validate_seed_receipt(receipt, config, config_sha256, seed)
                atomic_write_json(path_by_seed[seed], receipt)
                receipt_by_seed[seed] = receipt
                publish(seed)

    receipts = ordered_receipts()
    complete = len(receipts) == len(config["seeds"])
    result = aggregate_receipts(
        config,
        config_sha256,
        prerequisite,
        receipts,
        complete=complete,
    )
    atomic_write_json(out_path, result)
    return result


__all__ = [
    "ACTORS",
    "ATLAS_SCHEMA",
    "CLAIM_SCOPE",
    "CONFIG_SCHEMA",
    "CONTEXTS",
    "MAX_SEED_WORKERS",
    "SEED_SCHEMA",
    "TASK_FAMILY",
    "aggregate_receipts",
    "atomic_write_json",
    "canonical_bytes",
    "canonical_sha256",
    "load_config",
    "make_dataset",
    "predict_prototype",
    "run_atlas",
    "run_seed",
    "sha256_file",
    "validate_prerequisite",
    "validate_seed_receipt",
]
