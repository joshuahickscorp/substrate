
from __future__ import annotations

import copy
import datetime as dt
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256, sha256_file

VERIFICATION_SCHEMA = wave.VERIFICATION_SCHEMA
CLAIM_SCOPE = wave.VERIFICATION_CLAIM_SCOPE
MUTATION_NAMES = (
    "program_identity",
    "fresh_cycle_boundary",
    "compute_started",
    "capsule_count",
    "wave_route",
    "maximum_compute",
    "integration_binding",
    "synthetic_confirmation_claim",
)
MUTATION_COUNT = len(MUTATION_NAMES)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"verification authority must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"verification authority is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError("verification artifact is outside the repository") from exc


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(core):
        raise ValueError(f"{label} seal drifted")


def _root_from_result(value: Mapping[str, Any]) -> Path:
    waves = value.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("categorized-wave result has no wave bindings")
    categories = waves[0].get("categories") if isinstance(waves[0], Mapping) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError("categorized-wave result has no category bindings")
    path_value = categories[0].get("path") if isinstance(categories[0], Mapping) else None
    if not isinstance(path_value, str):
        raise ValueError("categorized-wave result category path is invalid")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("categorized-wave result category path is unsafe")
    path = (REPO_ROOT / relative).resolve()
    root = path.parents[2]
    if not root.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("categorized-wave result root escapes the repository")
    return root


def _reseal(value: dict[str, Any], field: str) -> None:
    value.pop(field, None)
    value[field] = canonical_sha256(value)


def _mutation_rejected(
    result: Mapping[str, Any],
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> bool:
    candidate = copy.deepcopy(dict(result))
    mutate(candidate)
    _reseal(candidate, "result_sha256")
    try:
        wave.validate_result(candidate, root=root)
    except (OSError, ValueError):
        return True
    return False


def _mutation_suite(result: Mapping[str, Any], root: Path) -> dict[str, Any]:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "program_identity",
            lambda candidate: candidate.__setitem__("program_id", "forged-program"),
        ),
        (
            "fresh_cycle_boundary",
            lambda candidate: candidate["grid"]["fresh_cycle_indices"].__setitem__(0, 11),
        ),
        (
            "compute_started",
            lambda candidate: candidate["execution"].__setitem__(
                "compute_started",
                not bool(candidate["execution"]["compute_started"]),
            ),
        ),
        (
            "capsule_count",
            lambda candidate: candidate["grid"].__setitem__("manifest_capsule_count", 58),
        ),
        (
            "wave_route",
            lambda candidate: candidate["waves"][0]["routing"][wave.CATEGORY_IDS[0]].__setitem__(
                "continue",
                not bool(candidate["waves"][0]["routing"][wave.CATEGORY_IDS[0]]["continue"]),
            ),
        ),
        (
            "maximum_compute",
            lambda candidate: candidate["execution"].__setitem__(
                "maximum_planned_serial_seconds",
                wave.planned_program_compute_seconds() + 1,
            ),
        ),
        (
            "integration_binding",
            lambda candidate: candidate["integration"].__setitem__(
                "path",
                "runs/generation1/forged/i1.json",
            ),
        ),
        (
            "synthetic_confirmation_claim",
            lambda candidate: candidate.__setitem__("ready_for_confirmatory_claim", True),
        ),
    ]
    if tuple(name for name, _mutate in mutations) != MUTATION_NAMES:
        raise ValueError("categorized-wave mutation inventory drifted")
    rows = [
        {"mutation": name, "rejected": _mutation_rejected(result, root, mutate)} for name, mutate in mutations
    ]
    rejected = sum(row["rejected"] is True for row in rows)
    return {
        "count": len(rows),
        "rejected": rejected,
        "all_rejected": rejected == len(rows),
        "cases": rows,
    }


def _recomputation_from_result(
    result: Mapping[str, Any],
) -> dict[str, int | float]:
    category_count = 0
    shard_count = 0
    pruned_routes = 0
    raw_receipt_count = 0
    observed_seconds = 0.0
    for row in result["waves"]:
        for binding in row["categories"]:
            path = (REPO_ROOT / binding["path"]).resolve()
            category = _read_object(path)
            category_count += 1
            shard_count += len(category["balanced_planning_shards"])
            raw_receipt_count += int(category["execution"]["executed_item_count"])
            observed_seconds += float(category["execution"]["observed_seconds"])
            if category["execution"]["eligible"] is False:
                pruned_routes += 1
    integration_path = (REPO_ROOT / result["integration"]["path"]).resolve()
    integration = _read_object(integration_path)
    shard_count += len(integration["balanced_planning_shards"])
    raw_receipt_count += int(integration["execution"]["executed_item_count"])
    observed_seconds += float(integration["execution"]["observed_seconds"])
    return {
        "gate_count": len(wave.GATE_IDS),
        "wave_count": len(wave.EPOCH_IDS),
        "category_artifact_count": category_count,
        "planning_shard_descriptor_count": shard_count,
        "pruned_category_route_count": pruned_routes,
        "raw_receipt_count": raw_receipt_count,
        "observed_seconds": observed_seconds,
        "maximum_raw_receipt_count": wave.MAXIMUM_RAW_RECEIPT_COUNT,
        "manifest_capsule_count": wave.CAPSULE_COUNT,
        "maximum_planned_serial_seconds": wave.planned_program_compute_seconds(),
        "maximum_planned_serial_hours": wave.planned_serial_hours(),
        "maximum_ideal_eight_worker_hours": wave.planned_ideal_eight_worker_hours(),
    }


def build_verification(result_path: Path = wave.DEFAULT_RESULT) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    if not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("categorized-wave result is outside the repository")
    result = _read_object(result_path)
    root = _root_from_result(result)
    wave.validate_result(result, root=root, replay_d1_raw=True)
    mutation_suite = _mutation_suite(result, root)
    if not mutation_suite["all_rejected"]:
        raise ValueError("categorized-wave semantic mutation suite did not reject every mutation")

    recomputation = _recomputation_from_result(result)
    core = {
        "schema": VERIFICATION_SCHEMA,
        "program_id": wave.PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "verified_at": _now(),
        "source": {
            "path": _relative(result_path),
            "file_sha256": sha256_file(result_path),
            "result_sha256": result["result_sha256"],
        },
        "checks": {
            "result_seal_valid": True,
            "five_d1_transition_gates_valid": True,
            "all_seven_wave_classifiers_valid": True,
            "six_categories_per_wave_valid": True,
            "dynamic_eight_worker_pool_valid": True,
            "balanced_planning_shards_valid": True,
            "null_safe_pruning_valid": True,
            "single_post_w07_i1_integration_valid": True,
            "fresh_mechanics_receipts_valid": True,
            "observed_execution_accounted": True,
            "d1_redesign_efficacy_not_executed": True,
            "mutation_suite_passed": True,
            "independent_generator_family_present": False,
        },
        "recomputation": recomputation,
        "mutation_suite": mutation_suite,
        "verification_complete": True,
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def validate_verification(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema",
        "program_id",
        "claim_scope",
        "verified_at",
        "source",
        "checks",
        "recomputation",
        "mutation_suite",
        "verification_complete",
        "independent_scientific_confirmation",
        "complete",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "verification_sha256",
    }:
        raise ValueError("categorized-wave verification fields drifted")
    _validate_seal(value, "verification_sha256", "categorized-wave verification")
    verified_at = value.get("verified_at")
    if not isinstance(verified_at, str):
        raise ValueError("categorized-wave verification timestamp is invalid")
    try:
        parsed = dt.datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("categorized-wave verification timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("categorized-wave verification timestamp is not UTC")
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "file_sha256",
        "result_sha256",
    }:
        raise ValueError("categorized-wave verification source fields drifted")
    path_value = source.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("categorized-wave verification source path is invalid")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("categorized-wave verification source path is unsafe")
    result_path = (REPO_ROOT / relative).resolve()
    if not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("categorized-wave verification source path escapes the repository")
    result = _read_object(result_path)
    root = _root_from_result(result)
    wave.validate_result(result, root=root)
    if source.get("file_sha256") != sha256_file(result_path) or source.get("result_sha256") != result.get(
        "result_sha256"
    ):
        raise ValueError("categorized-wave verification source bytes drifted")
    expected_checks = {
        "result_seal_valid": True,
        "five_d1_transition_gates_valid": True,
        "all_seven_wave_classifiers_valid": True,
        "six_categories_per_wave_valid": True,
        "dynamic_eight_worker_pool_valid": True,
        "balanced_planning_shards_valid": True,
        "null_safe_pruning_valid": True,
        "single_post_w07_i1_integration_valid": True,
        "fresh_mechanics_receipts_valid": True,
        "observed_execution_accounted": True,
        "d1_redesign_efficacy_not_executed": True,
        "mutation_suite_passed": True,
        "independent_generator_family_present": False,
    }
    recomputation = value.get("recomputation")
    expected_recomputation = _recomputation_from_result(result)
    expected_mutation_suite = _mutation_suite(result, root)
    if (
        value.get("schema") != VERIFICATION_SCHEMA
        or value.get("program_id") != wave.PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("checks") != expected_checks
        or recomputation != expected_recomputation
        or value.get("mutation_suite") != expected_mutation_suite
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("categorized-wave verification identity or recomputation drifted")


def verify(
    *,
    result_path: Path = wave.DEFAULT_RESULT,
    output: Path = wave.DEFAULT_VERIFICATION,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("categorized-wave verification is outside the repository")
    if output.is_file():
        existing = _read_object(output)
        validate_verification(existing)
        return existing
    value = build_verification(result_path)
    validate_verification(value)
    atomic_write_json(output, value)
    return value


__all__ = [
    "CLAIM_SCOPE",
    "MUTATION_COUNT",
    "MUTATION_NAMES",
    "VERIFICATION_SCHEMA",
    "build_verification",
    "validate_verification",
    "verify",
]
