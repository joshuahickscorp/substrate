
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT

from .generation1_competence_atlas import (
    ACTORS,
    ATLAS_SCHEMA,
    CLAIM_SCOPE,
    CONTEXTS,
    SEED_SCHEMA,
    TASK_FAMILY,
    atomic_write_json,
    canonical_sha256,
    load_config,
    make_dataset,
    predict_prototype,
    sha256_file,
    validate_prerequisite,
)

VERIFICATION_SCHEMA = "mop-generation1-competence-atlas-verification/v1"


def _read(path: Path | str, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _seal_ok(value: Mapping[str, Any], field: str) -> bool:
    observed = value.get(field)
    core = {key: item for key, item in value.items() if key != field}
    return observed == canonical_sha256(core)


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _accuracy(prediction: Sequence[int], truth: Sequence[int], mask: Sequence[bool]) -> float:
    selected = [index for index, include in enumerate(mask) if include]
    if not selected:
        return 0.0
    return sum(int(prediction[index] == truth[index]) for index in selected) / len(selected)


def _interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def _independent_context_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    truth = [int(value) for value in row["truth"]]
    context_ids = [int(value) for value in row["context_ids"]]
    predictions = {actor: [int(value) for value in row["predictions"][actor]] for actor in ACTORS}
    homogeneous = [[int(value) for value in prediction] for prediction in row["homogeneous_predictions"]]
    random_prediction = [int(value) for value in row["random_prediction"]]
    result: dict[str, Any] = {}
    for context_index, context in enumerate(CONTEXTS):
        mask = [value == context_index for value in context_ids]
        actor_accuracy = {actor: _accuracy(predictions[actor], truth, mask) for actor in ACTORS}
        oracle = [
            truth[index]
            if any(predictions[actor][index] == truth[index] for actor in ACTORS)
            else predictions[ACTORS[0]][index]
            for index in range(len(truth))
        ]
        homogeneous_oracle = [
            truth[index]
            if any(prediction[index] == truth[index] for prediction in homogeneous)
            else homogeneous[0][index]
            for index in range(len(truth))
        ]
        accepted = 0
        accepted_correct = 0
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
                accepted_correct += int(label == truth[index])
        total = sum(mask)
        result[context] = {
            "observation_count": total,
            "actor_accuracy": actor_accuracy,
            "best_single_accuracy": max(actor_accuracy.values()),
            "random_accuracy": _accuracy(random_prediction, truth, mask),
            "oracle_actor_accuracy": _accuracy(oracle, truth, mask),
            "homogeneous_copy_mean_accuracy": statistics.fmean(
                _accuracy(prediction, truth, mask) for prediction in homogeneous
            ),
            "homogeneous_oracle_accuracy": _accuracy(homogeneous_oracle, truth, mask),
            "abstention": {
                "coverage": 0.0 if total == 0 else accepted / total,
                "selective_accuracy": (None if accepted == 0 else accepted_correct / accepted),
                "accepted": accepted,
                "total": total,
                "vote_threshold": 3,
            },
        }
    return result


def _receipt_problems(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    expected_seed: int,
    *,
    regenerate_datasets: bool,
) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != SEED_SCHEMA:
        problems.append("seed schema drifted")
    if not _seal_ok(receipt, "seed_sha256"):
        problems.append("seed self-seal invalid")
    if receipt.get("campaign_id") != config["campaign_id"]:
        problems.append("seed campaign drifted")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        problems.append("seed claim scope drifted")
    if receipt.get("config_file_sha256") != config_sha256:
        problems.append("seed config authority drifted")
    if receipt.get("seed") != expected_seed or receipt.get("complete") is not True:
        problems.append("seed identity or completion drifted")
    if receipt.get("activation_allowed") is not False:
        problems.append("seed activation escaped")
    if receipt.get("scientific_promotion") is not False:
        problems.append("seed promotion escaped")
    rows = receipt.get("difficulties")
    if not isinstance(rows, list) or len(rows) != len(config["difficulty_separations"]):
        problems.append("seed difficulty inventory drifted")
        return problems
    n_test = int(config["dataset"]["n_test"])
    copies = int(config["training"]["homogeneous_copies"])
    dataset = config["dataset"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"difficulty {index} is not an object")
            continue
        if row.get("difficulty_index") != index:
            problems.append(f"difficulty {index} coordinate drifted")
        truth = row.get("truth")
        context_ids = row.get("context_ids")
        predictions = row.get("predictions")
        homogeneous = row.get("homogeneous_predictions")
        random_prediction = row.get("random_prediction")
        if not isinstance(truth, list) or len(truth) != n_test:
            problems.append(f"difficulty {index} truth length drifted")
            continue
        if not isinstance(context_ids, list) or len(context_ids) != n_test:
            problems.append(f"difficulty {index} context length drifted")
            continue
        if set(context_ids) != set(range(len(CONTEXTS))):
            problems.append(f"difficulty {index} context inventory drifted")
        if not isinstance(predictions, dict) or set(predictions) != set(ACTORS):
            problems.append(f"difficulty {index} actor inventory drifted")
            continue
        if any(
            not isinstance(predictions[actor], list) or len(predictions[actor]) != n_test for actor in ACTORS
        ):
            problems.append(f"difficulty {index} prediction length drifted")
            continue
        if (
            not isinstance(homogeneous, list)
            or len(homogeneous) != copies
            or any(
                not isinstance(prediction, list) or len(prediction) != n_test for prediction in homogeneous
            )
        ):
            problems.append(f"difficulty {index} homogeneous control drifted")
            continue
        if not isinstance(random_prediction, list) or len(random_prediction) != n_test:
            problems.append(f"difficulty {index} random control drifted")
            continue
        rebuilt = _independent_context_metrics(row)
        if row.get("context_metrics") != rebuilt:
            problems.append(f"difficulty {index} metrics do not rebuild")
        if regenerate_datasets:
            effective_seed = expected_seed + index * 1_000_003
            generated = make_dataset(
                effective_seed,
                int(dataset["n_train"]),
                n_test,
                int(dataset["n_classes"]),
                int(dataset["dim"]),
                float(config["difficulty_separations"][index]),
            )
            if _tensor_digest(*generated) != row.get("dataset_sha256"):
                problems.append(f"difficulty {index} generated dataset digest drifted")
            if [int(value) for value in generated[3].tolist()] != truth:
                problems.append(f"difficulty {index} truth does not match generated dataset")
            if [int(value) for value in generated[4].tolist()] != context_ids:
                problems.append(f"difficulty {index} contexts do not match generated dataset")
    return problems


def _independent_aggregate(
    config: Mapping[str, Any],
    config_sha256: str,
    prerequisite: Mapping[str, Any],
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    criteria = config["criteria"]
    chance = 1.0 / int(config["dataset"]["n_classes"])
    tensor: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    headroom_candidates: list[dict[str, Any]] = []
    for difficulty_index, separation in enumerate(config["difficulty_separations"]):
        for context in CONTEXTS:
            accuracies: dict[str, list[float]] = {actor: [] for actor in ACTORS}
            best_values: list[float] = []
            oracle_values: list[float] = []
            random_values: list[float] = []
            homogeneous_values: list[float] = []
            coverages: list[float] = []
            selective: list[float] = []
            for _, receipt in receipts:
                metrics = _independent_context_metrics(receipt["difficulties"][difficulty_index])[context]
                for actor in ACTORS:
                    accuracies[actor].append(float(metrics["actor_accuracy"][actor]))
                best_values.append(float(metrics["best_single_accuracy"]))
                oracle_values.append(float(metrics["oracle_actor_accuracy"]))
                random_values.append(float(metrics["random_accuracy"]))
                homogeneous_values.append(float(metrics["homogeneous_oracle_accuracy"]))
                coverages.append(float(metrics["abstention"]["coverage"]))
                value = metrics["abstention"]["selective_accuracy"]
                if value is not None:
                    selective.append(float(value))
            winner = max((statistics.fmean(values), actor) for actor, values in accuracies.items())[1]
            for actor in ACTORS:
                values = accuracies[actor]
                deltas = [
                    value - max(accuracies[other][seed_index] for other in ACTORS if other != actor)
                    for seed_index, value in enumerate(values)
                ]
                quality = _interval(values)
                advantage = _interval(deltas)
                off_ceiling = bool(
                    chance + float(criteria["above_chance_margin"])
                    <= float(quality["mean"])
                    <= float(criteria["off_ceiling_max_accuracy"])
                )
                favorable = sum(value > 0 for value in deltas) / len(deltas)
                candidate = bool(
                    actor == winner
                    and off_ceiling
                    and float(advantage["lo"]) > 0
                    and float(advantage["mean"]) >= float(criteria["min_niche_advantage"])
                    and favorable >= float(criteria["min_reproducible_fraction"])
                )
                row = {
                    "task_family": TASK_FAMILY,
                    "context": context,
                    "difficulty_index": difficulty_index,
                    "separation": float(separation),
                    "actor": actor,
                    "quality": quality,
                    "advantage_over_next_best": advantage,
                    "favorable_seed_fraction": favorable,
                    "off_ceiling": off_ceiling,
                    "niche_candidate": candidate,
                }
                tensor.append(row)
                if candidate:
                    candidates.append(row)
            headroom = _interval(
                [oracle - best for oracle, best in zip(oracle_values, best_values, strict=True)]
            )
            headroom_candidate = bool(
                float(headroom["lo"]) > 0
                and float(headroom["mean"]) >= float(criteria["min_oracle_headroom"])
            )
            control = {
                "task_family": TASK_FAMILY,
                "context": context,
                "difficulty_index": difficulty_index,
                "separation": float(separation),
                "best_single": _interval(best_values),
                "random": _interval(random_values),
                "homogeneous_oracle": _interval(homogeneous_values),
                "oracle_actor": _interval(oracle_values),
                "oracle_headroom_over_best_single": headroom,
                "oracle_headroom_candidate": headroom_candidate,
                "abstention_coverage": _interval(coverages),
                "abstention_selective_accuracy": _interval(selective),
            }
            controls.append(control)
            if headroom_candidate:
                headroom_candidates.append(control)
    actor_contexts: dict[str, set[str]] = {}
    for row in candidates:
        actor_contexts.setdefault(str(row["actor"]), set()).add(str(row["context"]))
    pairs = []
    actors = sorted(actor_contexts)
    for index, left in enumerate(actors):
        for right in actors[index + 1 :]:
            left_only = sorted(actor_contexts[left] - actor_contexts[right])
            right_only = sorted(actor_contexts[right] - actor_contexts[left])
            if left_only and right_only:
                pairs.append(
                    {
                        "left_actor": left,
                        "right_actor": right,
                        "left_only_contexts": left_only,
                        "right_only_contexts": right_only,
                    }
                )
    expected_cells = len(config["seeds"]) * len(config["difficulty_separations"])
    decision = {
        "bed_valid": bool(candidates and headroom_candidates),
        "off_ceiling_cell_count": sum(row["off_ceiling"] for row in tensor),
        "niche_candidate_count": len(candidates),
        "niche_actor_ids": sorted(actor_contexts),
        "oracle_headroom_cell_count": len(headroom_candidates),
        "disjoint_actor_pairs": pairs,
        "reproducible_context_disjoint_niches": bool(pairs),
        "ready_to_preregister_g1_c2": bool(candidates and headroom_candidates and pairs),
        "ready_to_train_dispatcher": False,
        "verdict": (
            "generated_bed_disjoint_niches_candidate_pending_independent_verification"
            if candidates and headroom_candidates and pairs
            else "generated_bed_complete_without_c2_license"
        ),
    }
    return {
        "grid": {
            "expected_seed_count": len(config["seeds"]),
            "completed_seed_count": len(receipts),
            "expected_seed_difficulty_cells": expected_cells,
            "completed_seed_difficulty_cells": len(receipts) * len(config["difficulty_separations"]),
        },
        "competence_tensor": tensor,
        "controls": controls,
        "decision": decision,
        "prerequisite": dict(prerequisite),
        "config_file_sha256": config_sha256,
    }


def _semantic_problems(
    atlas: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    prerequisite: Mapping[str, Any],
    *,
    regenerate_datasets: bool,
) -> list[str]:
    problems: list[str] = []
    if atlas.get("schema") != ATLAS_SCHEMA:
        problems.append("atlas schema drifted")
    if not _seal_ok(atlas, "atlas_sha256"):
        problems.append("atlas self-seal invalid")
    if atlas.get("claim_scope") != CLAIM_SCOPE:
        problems.append("atlas claim scope drifted")
    if atlas.get("activation_allowed") is not False:
        problems.append("atlas activation escaped")
    if atlas.get("scientific_promotion") is not False:
        problems.append("atlas promotion escaped")
    if atlas.get("complete") is not True or atlas.get("problems") != []:
        problems.append("atlas is not cleanly complete")
    expected_seeds = list(config["seeds"])
    if [receipt.get("seed") for _, receipt in receipts] != expected_seeds:
        problems.append("seed receipt order or inventory drifted")
    for (_, receipt), seed in zip(receipts, expected_seeds, strict=False):
        problems.extend(
            f"seed {seed}: {problem}"
            for problem in _receipt_problems(
                receipt,
                config,
                config_sha256,
                seed,
                regenerate_datasets=regenerate_datasets,
            )
        )
    if problems or len(receipts) != len(expected_seeds):
        return problems
    rebuilt = _independent_aggregate(config, config_sha256, prerequisite, receipts)
    for field in ("grid", "competence_tensor", "controls", "decision", "prerequisite"):
        if atlas.get(field) != rebuilt[field]:
            problems.append(f"independent {field} rebuild drifted")
    config_row = atlas.get("config")
    if not isinstance(config_row, dict) or config_row.get("file_sha256") != config_sha256:
        problems.append("atlas config binding drifted")
    return problems


def _mutation_suite(
    atlas: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    prerequisite: Mapping[str, Any],
) -> dict[str, Any]:
    mutations: list[tuple[str, dict[str, Any], list[tuple[str, Mapping[str, Any]]]]] = []

    def reseal(value: dict[str, Any], field: str) -> None:
        value[field] = canonical_sha256({key: item for key, item in value.items() if key != field})

    activation = copy.deepcopy(dict(atlas))
    activation["activation_allowed"] = True
    reseal(activation, "atlas_sha256")
    mutations.append(("activation_enabled", activation, list(receipts)))

    promotion = copy.deepcopy(dict(atlas))
    promotion["scientific_promotion"] = True
    reseal(promotion, "atlas_sha256")
    mutations.append(("scientific_promotion_enabled", promotion, list(receipts)))

    claim = copy.deepcopy(dict(atlas))
    claim["claim_scope"] = "natural-world-general-actor-niches"
    reseal(claim, "atlas_sha256")
    mutations.append(("claim_scope_escalated", claim, list(receipts)))

    config_drift = copy.deepcopy(dict(atlas))
    config_drift["config"]["file_sha256"] = "0" * 64
    reseal(config_drift, "atlas_sha256")
    mutations.append(("config_binding_drifted", config_drift, list(receipts)))

    missing = list(receipts[:-1])
    mutations.append(("seed_removed", copy.deepcopy(dict(atlas)), missing))

    prediction_rows = copy.deepcopy(list(receipts))
    prediction_receipt = dict(prediction_rows[0][1])
    prediction_receipt["difficulties"][0]["predictions"][ACTORS[0]].pop()
    reseal(prediction_receipt, "seed_sha256")
    prediction_rows[0] = (prediction_rows[0][0], prediction_receipt)
    mutations.append(("prediction_removed", copy.deepcopy(dict(atlas)), prediction_rows))

    actor_rows = copy.deepcopy(list(receipts))
    actor_receipt = dict(actor_rows[0][1])
    actor_receipt["difficulties"][0]["predictions"]["undeclared_actor"] = list(
        actor_receipt["difficulties"][0]["truth"]
    )
    reseal(actor_receipt, "seed_sha256")
    actor_rows[0] = (actor_rows[0][0], actor_receipt)
    mutations.append(("undeclared_actor_added", copy.deepcopy(dict(atlas)), actor_rows))

    decision = copy.deepcopy(dict(atlas))
    decision["decision"]["ready_to_train_dispatcher"] = True
    reseal(decision, "atlas_sha256")
    mutations.append(("dispatcher_activation_enabled", decision, list(receipts)))

    rows = []
    for name, mutated_atlas, mutated_receipts in mutations:
        problems = _semantic_problems(
            mutated_atlas,
            config,
            config_sha256,
            mutated_receipts,
            prerequisite,
            regenerate_datasets=False,
        )
        rows.append({"mutation": name, "rejected": bool(problems), "problems": problems})
    return {
        "count": len(rows),
        "rejected": sum(row["rejected"] for row in rows),
        "all_rejected": all(row["rejected"] for row in rows),
        "mutations": rows,
    }


def verify_atlas(
    config_path: Path | str,
    atlas_path: Path | str,
    out_path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    prerequisite = validate_prerequisite(config, repo_root=repo_root)
    atlas = _read(atlas_path, "competence atlas")
    receipt_rows = atlas.get("seed_receipts")
    receipts: list[tuple[str, Mapping[str, Any]]] = []
    problems: list[str] = []
    if not isinstance(receipt_rows, list):
        problems.append("atlas seed receipt index is invalid")
    else:
        for index, row in enumerate(receipt_rows):
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                problems.append(f"seed receipt index {index} is invalid")
                continue
            path = Path(str(row["path"]))
            source = path if path.is_absolute() else repo_root / path
            try:
                receipt = _read(source, f"seed receipt {index}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"seed receipt {index} unreadable: {type(exc).__name__}: {exc}")
                continue
            if row.get("seed_sha256") != receipt.get("seed_sha256"):
                problems.append(f"seed receipt {index} index seal drifted")
            receipts.append((str(row["path"]), receipt))
    if not problems:
        problems.extend(
            _semantic_problems(
                atlas,
                config,
                config_sha256,
                receipts,
                prerequisite,
                regenerate_datasets=True,
            )
        )
    canary: dict[str, Any] = {"performed": False, "matched": False}
    if receipts and not problems:
        first = receipts[0][1]["difficulties"][0]
        dataset = config["dataset"]
        generated = make_dataset(
            int(receipts[0][1]["seed"]),
            int(dataset["n_train"]),
            int(dataset["n_test"]),
            int(dataset["n_classes"]),
            int(dataset["dim"]),
            float(config["difficulty_separations"][0]),
        )
        prediction = predict_prototype(generated[0], generated[1], generated[2], int(dataset["n_classes"]))
        expected = [int(value) for value in first["predictions"]["prototype"]]
        matched = [int(value) for value in prediction.tolist()] == expected
        canary = {
            "performed": True,
            "actor": "prototype",
            "seed": receipts[0][1]["seed"],
            "difficulty_index": 0,
            "matched": matched,
            "prediction_sha256": canonical_sha256(expected),
        }
        if not matched:
            problems.append("fresh deterministic prototype canary did not reproduce")
    mutations = (
        _mutation_suite(atlas, config, config_sha256, receipts, prerequisite)
        if receipts
        else {"count": 0, "rejected": 0, "all_rejected": False, "mutations": []}
    )
    if not mutations["all_rejected"]:
        problems.append("mutation suite did not reject every claim or evidence mutation")
    verified_decision = dict(atlas.get("decision") or {})
    if problems:
        verified_decision["ready_to_preregister_g1_c2"] = False
    core = {
        "schema": VERIFICATION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config": {"path": str(config_path), "file_sha256": config_sha256},
        "atlas": {
            "path": str(atlas_path),
            "file_sha256": sha256_file(atlas_path),
            "atlas_sha256": atlas.get("atlas_sha256"),
        },
        "checks": {
            "atlas_self_seal": _seal_ok(atlas, "atlas_sha256"),
            "seed_count_exact": len(receipts) == len(config["seeds"]),
            "all_seed_receipts_rebuilt": not problems,
            "all_generated_datasets_reproduced": not problems,
            "fresh_actor_canary": canary.get("matched") is True,
            "promotion_blocked": atlas.get("scientific_promotion") is False,
            "activation_blocked": atlas.get("activation_allowed") is False,
            "dispatcher_training_blocked": (atlas.get("decision") or {}).get("ready_to_train_dispatcher")
            is False,
        },
        "fresh_canary": canary,
        "mutation_suite": mutations,
        "verified_decision": verified_decision,
        "verification_complete": not problems,
        "problems": problems,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = {
        **core,
        "verification_sha256": canonical_sha256(core),
    }
    atomic_write_json(out_path, result)
    return result


__all__ = ["VERIFICATION_SCHEMA", "verify_atlas"]
