#!/usr/bin/env python3
"""Freeze the executable Generation-1 corpus program from the current local authority."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies.generation1_cognitive_corpus import eligible_experiment_ids, load_config
from mop.studio.generation1_supervisor import (
    CAPSULE_SCHEMA,
    PROGRAM_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
)
from mop.studio.local_throttle import load_policy

DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_empirical_program_v2.json"
CORPUS_CONFIG = REPO_ROOT / "configs/experiment/generation1_cognitive_corpus.json"
RESOURCE_CANARY_RECEIPT = REPO_ROOT / "proof/GENERATION1_RESOURCE_CANARY.json"
PROGRAM_ID = "generation1-empirical-cognitive-corpus-v2"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"

EXPECTED_CONFIG_SCHEMA = "mop-generation1-cognitive-corpus-config/v2"
EXPECTED_CAMPAIGN_ID = "generation1-empirical-cognitive-corpus-v2"
EXPECTED_RESULT_TAG = "generation1-exploratory-corpus-v2"
RESOURCE_CANARY_SCHEMA = "mop-generation1-resource-canary/v1"
EXPECTED_SEED_COUNT = 24
EXPECTED_EXPERIMENT_COUNT = 128
EXPECTED_MODE_COUNTS = {"fixed": 1, "mechanics": 2, "varied": 125}
EXPECTED_EVIDENCE_COUNTS = {
    "fixed_case_noninferential": 1,
    "inferential": 125,
    "mechanics_noninferential": 2,
}
EXPECTED_EFFECTIVE_EXECUTIONS = 3003
ADVANCED_VERIFIER_CHECKS = (
    "all_attempt_receipts_valid",
    "all_cell_authorities_valid",
    "seed_authority_exact",
    "no_pseudoreplication",
    "independent_summary_match",
    "directional_inference_fail_closed",
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _authority(path: Path | str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve() if not isinstance(path, Path) else path.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"authority must be a regular file: {source}")
    return {"path": _relative(source), "sha256": sha256_file(source)}


def _iter_authority_files() -> Iterable[Path]:
    roots = (
        REPO_ROOT / "src/mop",
        REPO_ROOT / "configs/experiment",
        REPO_ROOT / "registry",
    )
    for root in roots:
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ):
                yield path
    extras = (
        "pyproject.toml",
        "uv.lock",
        "configs/local_execution_throttle.yaml",
        "scripts/build_generation1_program.py",
        "scripts/build_generation1_injection.py",
        "scripts/mop_generation1_campaign.py",
        "scripts/generation1_cognitive_corpus.py",
        "scripts/generation1_resource_canary.py",
        "scripts/verify_generation1_cognitive_corpus.py",
        "scripts/build_generation1_report.py",
        "scripts/run_escs_x1_dispatch.py",
        "scripts/run_ecology_scaffold_batteries.py",
        "scripts/verify_ecology_scaffold_batteries.py",
        "scripts/run_integrity_scaffold_drills.py",
        "scripts/verify_integrity_scaffold_drills.py",
        "scripts/run_material_twin_batteries.py",
        "scripts/verify_material_twin_batteries.py",
        "scripts/run_integration_broadcast.py",
        "scripts/verify_integration_broadcast.py",
        "scripts/run_sensing_scaffold.py",
        "scripts/verify_sensing_scaffold.py",
        "scripts/run_escs_g0_formation_study.py",
        "scripts/p7_action_world_model_preflight.py",
        "scripts/p9_causal_monitoring_preflight.py",
        "proof/ESCS_X0_EVENT_FORMATION.verification.json",
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json",
        "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
        "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
        "proof/GENERATION1_RESOURCE_CANARY.json",
    )
    for value in extras:
        yield REPO_ROOT / value


def _artifact(
    path: str,
    schema: str,
    fields: dict[str, Any],
    seal_field: str | None,
) -> dict[str, Any]:
    return {"path": path, "schema": schema, "fields": fields, "seal_field": seal_field}


def _resources(
    *,
    lane: str,
    cores: int,
    memory_gb: float,
    write_gb: float,
    wall_minutes: int,
    marker: str,
    basis: str,
) -> dict[str, Any]:
    return {
        "lane": lane,
        "accelerator": "none",
        "cpu_cores": cores,
        "estimated_unified_memory_gb": memory_gb,
        "estimated_mps_gb": 0.0,
        "resource_basis": basis,
        "forecast_write_gb": write_gb,
        "atomic_write_gb": min(0.1, write_gb),
        "wall_minutes": wall_minutes,
        "process_marker": marker,
    }


def _capsule(
    *,
    capsule_id: str,
    kind: str,
    priority: int,
    depends_on: list[str],
    command: list[str],
    resources: dict[str, Any],
    artifacts: list[dict[str, Any]],
    authorities: list[str],
) -> dict[str, Any]:
    core = {
        "schema": CAPSULE_SCHEMA,
        "id": capsule_id,
        "kind": kind,
        "priority": priority,
        "depends_on": depends_on,
        "command": command,
        "cwd": ".",
        "environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        },
        "resources": resources,
        "artifacts": artifacts,
        "authorities": [_authority(path) for path in authorities],
    }
    return {**core, "capsule_sha256": canonical_sha256(core)}


def _python(script: str, *arguments: str) -> list[str]:
    return [".venv/bin/python", script, *arguments]


def _repository_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be repository-relative")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path


def _validate_snapshot(snapshot: object, label: str) -> set[str]:
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label} must be an object")
    core = dict(snapshot)
    declared = core.pop("aggregate_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise ValueError(f"{label} aggregate self-seal is invalid")
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label}.files must be a nonempty list")
    observed: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise ValueError(f"{label}.files[{index}] must be an object")
        path = _repository_file(row.get("path"), f"{label}.files[{index}].path")
        relative = _relative(path)
        if relative in observed:
            raise ValueError(f"{label} repeats source authority {relative}")
        if row.get("sha256") != sha256_file(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError(f"{label} source authority drifted: {relative}")
        observed.add(relative)
    return observed


def _campaign_profile(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != EXPECTED_CONFIG_SCHEMA:
        raise ValueError(f"Generation-1 config must use {EXPECTED_CONFIG_SCHEMA}")
    if config.get("campaign_id") != EXPECTED_CAMPAIGN_ID:
        raise ValueError(f"Generation-1 campaign_id must be {EXPECTED_CAMPAIGN_ID}")
    if config.get("result_tag") != EXPECTED_RESULT_TAG:
        raise ValueError(f"Generation-1 result_tag must be {EXPECTED_RESULT_TAG}")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != EXPECTED_SEED_COUNT:
        raise ValueError(f"Generation-1 v2 requires exactly {EXPECTED_SEED_COUNT} outer seeds")
    experiment_ids = eligible_experiment_ids(config)
    if len(experiment_ids) != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError(
            f"Generation-1 v2 requires exactly {EXPECTED_EXPERIMENT_COUNT} eligible experiments"
        )
    policy = config.get("seed_authority")
    if not isinstance(policy, dict):
        raise ValueError("Generation-1 v2 requires an explicit seed-authority policy")
    eligible = set(experiment_ids)
    fixed_ids = set(policy.get("fixed_case_experiment_ids") or [])
    mechanics_ids = set(policy.get("mechanics_only_experiment_ids") or [])
    if not fixed_ids <= eligible or not mechanics_ids <= eligible or fixed_ids & mechanics_ids:
        raise ValueError("fixed and mechanics execute-once classes must be disjoint eligible IDs")
    varied_ids = eligible - fixed_ids - mechanics_ids
    mode_counts = {
        "fixed": len(fixed_ids),
        "mechanics": len(mechanics_ids),
        "varied": len(varied_ids),
    }
    if mode_counts != EXPECTED_MODE_COUNTS:
        raise ValueError(
            f"Generation-1 v2 seed-mode census drifted: {mode_counts!r} != "
            f"{EXPECTED_MODE_COUNTS!r}"
        )
    effective_executions = (
        len(varied_ids) * len(seeds) + len(fixed_ids) + len(mechanics_ids)
    )
    if effective_executions != EXPECTED_EFFECTIVE_EXECUTIONS:
        raise ValueError(
            f"Generation-1 v2 effective execution count drifted: {effective_executions}"
        )
    return {
        "experiment_ids": experiment_ids,
        "mode_counts": mode_counts,
        "evidence_class_counts": dict(EXPECTED_EVIDENCE_COUNTS),
        "execute_once_ids": sorted(fixed_ids | mechanics_ids),
        "effective_executions": effective_executions,
    }


def _load_resource_canary(config: dict[str, Any]) -> dict[str, Any]:
    path = RESOURCE_CANARY_RECEIPT.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"sealed Generation-1 resource canary is missing: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("sealed Generation-1 resource canary is not valid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != RESOURCE_CANARY_SCHEMA:
        raise ValueError(f"Generation-1 resource canary must use {RESOURCE_CANARY_SCHEMA}")
    core = dict(receipt)
    declared = core.pop("receipt_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise ValueError("Generation-1 resource canary self-seal is invalid")
    preflight = receipt.get("preflight")
    if not isinstance(preflight, dict):
        raise ValueError("Generation-1 resource canary preflight is absent")
    preflight_core = dict(preflight)
    preflight_seal = preflight_core.pop("preflight_sha256", None)
    if not isinstance(preflight_seal, str) or preflight_seal != canonical_sha256(preflight_core):
        raise ValueError("Generation-1 resource canary preflight self-seal is invalid")
    if preflight.get("launch_authorized") is not True or preflight.get("problems") != []:
        raise ValueError("Generation-1 resource canary preflight was not cleanly authorized")
    if (
        preflight.get("required_config_schema") != EXPECTED_CONFIG_SCHEMA
        or preflight.get("observed_config_schema") != EXPECTED_CONFIG_SCHEMA
    ):
        raise ValueError("Generation-1 resource canary did not observe the v2 corpus config")
    configuration = receipt.get("configuration")
    recommendation = receipt.get("recommendation")
    measurements = receipt.get("measurements")
    source_authority = receipt.get("source_authority")
    workers = receipt.get("workers")
    if not all(
        isinstance(value, dict)
        for value in (configuration, recommendation, measurements, source_authority)
    ) or not isinstance(workers, list):
        raise ValueError("Generation-1 resource canary evidence structure is incomplete")
    assert isinstance(configuration, dict)
    assert isinstance(recommendation, dict)
    assert isinstance(measurements, dict)
    assert isinstance(source_authority, dict)
    if (
        receipt.get("complete") is not True
        or receipt.get("scientific_promotion") is not False
        or receipt.get("orchestration_problems") != []
        or recommendation.get("eligible") is not True
        or recommendation.get("scientific_promotion") is not False
    ):
        raise ValueError("Generation-1 resource canary did not complete as clean resource evidence")
    max_workers = recommendation.get("recommended_max_workers")
    memory_gb = recommendation.get("recommended_estimated_unified_memory_gb")
    memory_bytes = recommendation.get("recommended_estimated_unified_memory_bytes")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers <= 0
        or isinstance(memory_gb, bool)
        or not isinstance(memory_gb, int | float)
        or float(memory_gb) <= 0.0
        or isinstance(memory_bytes, bool)
        or not isinstance(memory_bytes, int)
        or memory_bytes != int(float(memory_gb) * 1_000_000_000)
    ):
        raise ValueError("Generation-1 resource canary recommendation is invalid")
    batch = configuration.get("batch")
    if (
        configuration.get("max_workers") != max_workers
        or configuration.get("result_tag") != config["result_tag"]
        or configuration.get("outer_seed") != config["seeds"][0]
        or not isinstance(batch, list)
        or len(batch) != max_workers
        or any(not isinstance(item, str) for item in batch)
        or len(set(batch)) != len(batch)
        or not set(batch) <= set(eligible_experiment_ids(config))
        or configuration.get("batch_sha256") != canonical_sha256(batch)
    ):
        raise ValueError("Generation-1 resource canary configuration is not the frozen v2 batch")
    worker_ids: list[str] = []
    for row in workers:
        if not isinstance(row, dict):
            raise ValueError("Generation-1 resource canary worker rows must be objects")
        worker_id = row.get("experiment_id")
        manifest = row.get("manifest")
        if (
            not isinstance(worker_id, str)
            or row.get("outcome") != "ok"
            or not isinstance(manifest, dict)
            or manifest.get("valid") is not True
        ):
            raise ValueError("Generation-1 resource canary did not validate every measured worker")
        worker_ids.append(worker_id)
    if (
        len(workers) != max_workers
        or sorted(worker_ids) != sorted(batch)
        or receipt.get("outcome_counts") != {"ok": max_workers}
    ):
        raise ValueError("Generation-1 resource canary did not validate every measured worker")
    aggregate_peak = measurements.get("aggregate_process_tree_peak_rss_bytes")
    individual_peak = recommendation.get("observed_max_individual_worker_rss_bytes")
    if (
        isinstance(aggregate_peak, bool)
        or not isinstance(aggregate_peak, int)
        or aggregate_peak <= 0
        or recommendation.get("observed_aggregate_process_tree_peak_rss_bytes")
        != aggregate_peak
        or isinstance(individual_peak, bool)
        or not isinstance(individual_peak, int)
        or individual_peak <= 0
        or measurements.get("runtime_safety_problems") != []
    ):
        raise ValueError("Generation-1 resource canary measurements are incomplete or unsafe")
    if (
        source_authority.get("stable") is not True
        or source_authority.get("before") != source_authority.get("after")
        or preflight.get("source_snapshot") != source_authority.get("before")
    ):
        raise ValueError("Generation-1 resource canary source authority was not stable")
    frozen_snapshot = source_authority.get("before")
    observed_sources = _validate_snapshot(
        frozen_snapshot, "resource_canary.source_authority.before"
    )
    assert isinstance(frozen_snapshot, dict)
    if frozen_snapshot.get("seed_policy_sha256") != canonical_sha256(config["seed_authority"]):
        raise ValueError("Generation-1 resource canary seed-policy authority drifted")
    required_sources = {
        _relative(CORPUS_CONFIG),
        "scripts/generation1_cognitive_corpus.py",
        "scripts/generation1_resource_canary.py",
        "src/mop/studies/generation1_cognitive_corpus.py",
        "src/mop/studies/generation1_resource_canary.py",
        "src/mop/harness/runner.py",
        "src/mop/experiments/__init__.py",
        "src/mop/studio/local_throttle.py",
        "configs/local_execution_throttle.yaml",
    }
    if not required_sources <= observed_sources:
        raise ValueError(
            "Generation-1 resource canary did not bind every required corpus source authority"
        )
    margin_rule = recommendation.get("margin_rule")
    scaling_boundary = recommendation.get("scaling_boundary")
    if (
        not isinstance(margin_rule, str)
        or not margin_rule.strip()
        or not isinstance(scaling_boundary, str)
        or not scaling_boundary.strip()
    ):
        raise ValueError("Generation-1 resource canary recommendation rationale is absent")
    return {
        "path": _relative(path),
        "receipt_sha256": declared,
        "file_sha256": sha256_file(path),
        "max_workers": max_workers,
        "memory_gb": float(memory_gb),
        "memory_bytes": memory_bytes,
        "aggregate_peak_rss_bytes": aggregate_peak,
        "individual_peak_rss_bytes": individual_peak,
        "margin_rule": margin_rule,
        "scaling_boundary": scaling_boundary,
    }


def build_program() -> dict[str, Any]:
    corpus_config = load_config(CORPUS_CONFIG)
    campaign_profile = _campaign_profile(corpus_config)
    experiment_count = len(campaign_profile["experiment_ids"])
    resource_canary = _load_resource_canary(corpus_config)
    x1_authority_path = REPO_ROOT / "proof/ESCS_X1_DISPATCH.implementation-authority.json"
    x1_authority = json.loads(x1_authority_path.read_text(encoding="utf-8"))
    x1_manifest_sha256 = x1_authority.get("manifest_sha256")
    if (
        not isinstance(x1_manifest_sha256, str)
        or len(x1_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in x1_manifest_sha256)
    ):
        raise ValueError("X1 implementation authority has no valid manifest_sha256")
    x1_authority_arguments = (
        "--implementation-authority",
        "proof/ESCS_X1_DISPATCH.implementation-authority.json",
        "--implementation-authority-sha256",
        x1_manifest_sha256,
        "--edcm-receipt",
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
        "--edcm-verification",
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json",
    )
    capsules: list[dict[str, Any]] = []
    quick_basis = (
        "existing deterministic mechanics driver; bounded CPU work, fresh output namespace, "
        "and independent verifier where available"
    )

    def quick_resources(marker: str, wall: int = 60) -> dict[str, Any]:
        return _resources(
            lane="cpu",
            cores=1,
            memory_gb=4.0,
            write_gb=0.25,
            wall_minutes=wall,
            marker=marker,
            basis=quick_basis,
        )

    capsules.append(
        _capsule(
            capsule_id="g1_gen0_x1_invalid_bed_closeout",
            kind="exploratory",
            priority=0,
            depends_on=[],
            command=_python("scripts/run_escs_x1_dispatch.py", *x1_authority_arguments),
            resources=quick_resources("run_escs_x1_dispatch.py"),
            artifacts=[
                _artifact(
                    "proof/ESCS_X1_DISPATCH.json",
                    "mop-escs-x1-receipt/v1",
                    {
                        "execution_status": "complete",
                        "all_ok": True,
                        "aggregate.terminal_route": "invalid_bed",
                        "candidate_activation_enabled": False,
                        "scientific_promotion": False,
                    },
                    "receipt_sha256",
                )
            ],
            authorities=[
                "scripts/run_escs_x1_dispatch.py",
                "src/mop/studies/escs_x1_dispatch.py",
                "configs/experiment/escs_x1_dispatch.json",
                "proof/ESCS_X1_DISPATCH.implementation-authority.json",
                "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
                "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json",
            ],
        )
    )
    capsules.append(
        _capsule(
            capsule_id="g1_gen0_x1_invalid_bed_verify",
            kind="verifier",
            priority=1,
            depends_on=["g1_gen0_x1_invalid_bed_closeout"],
            command=_python(
                "scripts/run_escs_x1_dispatch.py",
                *x1_authority_arguments,
                "--verify",
                "proof/ESCS_X1_DISPATCH.json",
            ),
            resources=quick_resources("run_escs_x1_dispatch.py"),
            artifacts=[
                _artifact(
                    "proof/ESCS_X1_DISPATCH.verification.json",
                    "mop-escs-x1-verification-artifact/v1",
                    {
                        "verification.terminal_route": "invalid_bed",
                        "candidate_activation_enabled": False,
                        "scientific_promotion": False,
                    },
                    "verification_artifact_sha256",
                )
            ],
            authorities=[
                "scripts/run_escs_x1_dispatch.py",
                "src/mop/studies/escs_x1_dispatch.py",
                "configs/experiment/escs_x1_dispatch.json",
                "proof/ESCS_X1_DISPATCH.implementation-authority.json",
                "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
                "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json",
            ],
        )
    )

    mechanics_root = "runs/generation1/mechanics"
    mechanics_pairs: list[dict[str, Any]] = [
        {
            "name": "ecology",
            "run_script": "scripts/run_ecology_scaffold_batteries.py",
            "verify_script": "scripts/verify_ecology_scaffold_batteries.py",
            "run_args": ["--out", f"{mechanics_root}/ecology.json"],
            "verify_args": [
                f"{mechanics_root}/ecology.json",
                "--report",
                f"{mechanics_root}/ecology.verification.json",
            ],
            "run_schema": "mop-ecology-scaffold-battery-run/v1",
            "verify_schema": "mop-ecology-scaffold-independent-verifier/v1",
            "run_fields": {"status": "programmatic-results-awaiting-independent-verification"},
            "verify_fields": {"verified": True, "all_mutations_rejected": True, "errors": []},
            "run_seal": "payload_sha256",
        },
        {
            "name": "integrity",
            "run_script": "scripts/run_integrity_scaffold_drills.py",
            "verify_script": "scripts/verify_integrity_scaffold_drills.py",
            "run_args": ["--out", f"{mechanics_root}/integrity.json"],
            "verify_args": [
                f"{mechanics_root}/integrity.json",
                "--report",
                f"{mechanics_root}/integrity.verification.json",
            ],
            "run_schema": "mop-integrity-scaffold-run/v1",
            "verify_schema": "mop-integrity-scaffold-independent-verifier/v1",
            "run_fields": {"status": "mechanics-pass"},
            "verify_fields": {"verified": True, "all_mutations_rejected": True, "errors": []},
            "run_seal": "payload_sha256",
        },
        {
            "name": "material_twin",
            "run_script": "scripts/run_material_twin_batteries.py",
            "verify_script": "scripts/verify_material_twin_batteries.py",
            "run_args": ["--out", f"{mechanics_root}/material_twin.json"],
            "verify_args": [
                f"{mechanics_root}/material_twin.json",
                "--report",
                f"{mechanics_root}/material_twin.verification.json",
            ],
            "run_schema": "mop-material-twin-batteries/v1",
            "verify_schema": "mop-material-twin-independent-verifier/v1",
            "run_fields": {"status": "complete"},
            "verify_fields": {"verified": True, "all_mutations_rejected": True, "errors": []},
            "run_seal": "payload_sha256",
        },
    ]
    previous = "g1_gen0_x1_invalid_bed_verify"
    priority = 10
    for pair in mechanics_pairs:
        run_id = f"g1_{pair['name']}_fresh_run"
        verify_id = f"g1_{pair['name']}_fresh_verify"
        run_output = pair["run_args"][-1]
        verify_output = pair["verify_args"][-1]
        capsules.append(
            _capsule(
                capsule_id=run_id,
                kind="corpus",
                priority=priority,
                depends_on=[previous],
                command=_python(pair["run_script"], *pair["run_args"]),
                resources=quick_resources(Path(pair["run_script"]).name),
                artifacts=[
                    _artifact(
                        run_output,
                        pair["run_schema"],
                        pair["run_fields"],
                        pair["run_seal"],
                    )
                ],
                authorities=[pair["run_script"]],
            )
        )
        capsules.append(
            _capsule(
                capsule_id=verify_id,
                kind="verifier",
                priority=priority + 1,
                depends_on=[run_id],
                command=_python(pair["verify_script"], *pair["verify_args"]),
                resources=quick_resources(Path(pair["verify_script"]).name),
                artifacts=[
                    _artifact(
                        verify_output,
                        pair["verify_schema"],
                        pair["verify_fields"],
                        None,
                    )
                ],
                authorities=[pair["verify_script"]],
            )
        )
        previous = verify_id
        priority += 2

    structured_pairs: list[dict[str, str]] = [
        {
            "name": "broadcast",
            "run_script": "scripts/run_integration_broadcast.py",
            "verify_script": "scripts/verify_integration_broadcast.py",
            "config": "configs/experiment/integration_broadcast_runs.yaml",
            "run_schema": "mop-integration-broadcast-run/v1",
            "verify_schema": "mop-integration-broadcast-verifier/v1",
        },
        {
            "name": "sensing",
            "run_script": "scripts/run_sensing_scaffold.py",
            "verify_script": "scripts/verify_sensing_scaffold.py",
            "config": "configs/experiment/sensing_scaffold_runs.yaml",
            "run_schema": "mop-sensing-scaffold-run/v1",
            "verify_schema": "mop-sensing-scaffold-verifier/v1",
        },
    ]
    for structured_pair in structured_pairs:
        run_id = f"g1_{structured_pair['name']}_fresh_run"
        verify_id = f"g1_{structured_pair['name']}_fresh_verify"
        run_output = f"{mechanics_root}/{structured_pair['name']}.json"
        verify_output = f"{mechanics_root}/{structured_pair['name']}.verification.json"
        config_argument = str((REPO_ROOT / structured_pair["config"]).resolve())
        capsules.append(
            _capsule(
                capsule_id=run_id,
                kind="corpus",
                priority=priority,
                depends_on=[previous],
                command=_python(
                    structured_pair["run_script"],
                    "--config",
                    config_argument,
                    "--out",
                    run_output,
                ),
                resources=quick_resources(Path(structured_pair["run_script"]).name),
                artifacts=[
                    _artifact(
                        run_output,
                        structured_pair["run_schema"],
                        {"scientific_capability_claim": False},
                        "payload_sha256",
                    )
                ],
                authorities=[structured_pair["run_script"], structured_pair["config"]],
            )
        )
        capsules.append(
            _capsule(
                capsule_id=verify_id,
                kind="verifier",
                priority=priority + 1,
                depends_on=[run_id],
                command=_python(
                    structured_pair["verify_script"],
                    "--run",
                    run_output,
                    "--config",
                    config_argument,
                    "--out",
                    verify_output,
                ),
                resources=quick_resources(Path(structured_pair["verify_script"]).name),
                artifacts=[
                    _artifact(
                        verify_output,
                        structured_pair["verify_schema"],
                        {"all_ok": True, "problems": [], "scientific_capability_claim": False},
                        "payload_sha256",
                    )
                ],
                authorities=[structured_pair["verify_script"], structured_pair["config"]],
            )
        )
        previous = verify_id
        priority += 2

    g0_run = f"{mechanics_root}/g0_formation.json"
    g0_verify = f"{mechanics_root}/g0_formation.verification.json"
    capsules.append(
        _capsule(
            capsule_id="g1_g0_formation_fresh_run",
            kind="corpus",
            priority=priority,
            depends_on=[previous],
            command=_python(
                "scripts/run_escs_g0_formation_study.py",
                "run",
                "--config",
                "configs/experiment/escs_g0_formation_study.json",
                "--out",
                g0_run,
            ),
            resources=quick_resources("run_escs_g0_formation_study.py"),
            artifacts=[
                _artifact(
                    g0_run,
                    "mop-escs-g0-formation-study-receipt/v1",
                    {
                        "all_ok": True,
                        "counterfactual_only": True,
                        "activation_enabled": False,
                        "scientific_promotion_allowed": False,
                    },
                    "receipt_sha256",
                )
            ],
            authorities=[
                "scripts/run_escs_g0_formation_study.py",
                "src/mop/studies/escs_g0_formation_study.py",
                "configs/experiment/escs_g0_formation_study.json",
            ],
        )
    )
    capsules.append(
        _capsule(
            capsule_id="g1_g0_formation_fresh_verify",
            kind="verifier",
            priority=priority + 1,
            depends_on=["g1_g0_formation_fresh_run"],
            command=_python(
                "scripts/run_escs_g0_formation_study.py",
                "verify",
                "--config",
                "configs/experiment/escs_g0_formation_study.json",
                "--receipt",
                g0_run,
                "--out",
                g0_verify,
            ),
            resources=quick_resources("run_escs_g0_formation_study.py"),
            artifacts=[
                _artifact(
                    g0_verify,
                    "mop-escs-g0-formation-study-verification/v1",
                    {
                        "all_ok": True,
                        "counterfactual_only": True,
                        "activation_enabled": False,
                        "scientific_promotion_allowed": False,
                    },
                    "verification_sha256",
                )
            ],
            authorities=[
                "scripts/run_escs_g0_formation_study.py",
                "src/mop/studies/escs_g0_formation_study.py",
                "configs/experiment/escs_g0_formation_study.json",
            ],
        )
    )
    previous = "g1_g0_formation_fresh_verify"
    priority += 2

    for name, script, config_path, schema in (
        (
            "p7_action_world",
            "scripts/p7_action_world_model_preflight.py",
            "configs/experiment/p7_action_world_model_preflight.yaml",
            "mop-p7-action-world-model-preflight/v1",
        ),
        (
            "p9_causal_monitor",
            "scripts/p9_causal_monitoring_preflight.py",
            "configs/experiment/p9_causal_monitoring_preflight.yaml",
            "mop-p9-causal-monitoring-preflight/v1",
        ),
    ):
        output = f"{mechanics_root}/{name}.json"
        capsule_id = f"g1_{name}_fresh_run"
        capsules.append(
            _capsule(
                capsule_id=capsule_id,
                kind="corpus",
                priority=priority,
                depends_on=[previous],
                command=_python(script, "--config", config_path, "--out", output),
                resources=quick_resources(Path(script).name),
                artifacts=[
                    _artifact(
                        output,
                        schema,
                        {"status": "mechanics-pass", "all_mechanics_ok": True},
                        None,
                    )
                ],
                authorities=[script, config_path],
            )
        )
        previous = capsule_id
        priority += 1

    seed_capsules: list[str] = []
    seed_dependency = previous
    first_outer_seed = int(corpus_config["seeds"][0])
    canary_basis = (
        f"sealed resource canary {resource_canary['path']} "
        f"(receipt {resource_canary['receipt_sha256']}, file {resource_canary['file_sha256']}) "
        f"cleanly completed the exact {resource_canary['max_workers']}-worker batch with "
        f"aggregate process-tree peak {resource_canary['aggregate_peak_rss_bytes']} bytes and "
        f"maximum individual-worker peak {resource_canary['individual_peak_rss_bytes']} bytes; "
        f"its frozen recommendation is {resource_canary['memory_gb']} GB under "
        f"{resource_canary['margin_rule']} ({resource_canary['scaling_boundary']}); the v2 corpus "
        f"contains {campaign_profile['mode_counts']['varied']} varied classes x "
        f"{len(corpus_config['seeds'])} outer seeds + "
        f"{campaign_profile['mode_counts']['fixed']} fixed class + "
        f"{campaign_profile['mode_counts']['mechanics']} mechanics classes = "
        f"{campaign_profile['effective_executions']} actual executions, while execute-once "
        "references preserve 128-class coverage in every seed receipt"
    )
    for index, seed in enumerate(corpus_config["seeds"]):
        capsule_id = f"g1_cognitive_seed_{seed}"
        seed_capsules.append(capsule_id)
        receipt = f"runs/generation1/cognitive_corpus/seed_{seed}/seed_receipt.json"
        skipped_execute_once = (
            [] if seed == first_outer_seed else campaign_profile["execute_once_ids"]
        )
        capsules.append(
            _capsule(
                capsule_id=capsule_id,
                kind="corpus",
                priority=100 + index,
                depends_on=[seed_dependency],
                command=_python(
                    "scripts/generation1_cognitive_corpus.py",
                    "run-seed",
                    "--config",
                    "configs/experiment/generation1_cognitive_corpus.json",
                    "--run-root",
                    "runs/generation1/cognitive_corpus",
                    "--seed",
                    str(seed),
                    "--out",
                    receipt,
                    "--max-workers",
                    str(resource_canary["max_workers"]),
                    "--timeout-seconds",
                    "1200",
                    "--wall-seconds",
                    "17400",
                ),
                resources=_resources(
                    lane="cpu",
                    cores=int(resource_canary["max_workers"]),
                    memory_gb=float(resource_canary["memory_gb"]),
                    write_gb=0.5,
                    wall_minutes=300,
                    marker="generation1_cognitive_corpus.py",
                    basis=canary_basis,
                ),
                artifacts=[
                    _artifact(
                        receipt,
                        "mop-generation1-cognitive-seed/v2",
                        {
                            "campaign_id": EXPECTED_CAMPAIGN_ID,
                            "seed": seed,
                            "config.sha256": sha256_file(CORPUS_CONFIG),
                            "eligible_count": experiment_count,
                            "complete_count": experiment_count,
                            "skipped_execute_once_ids": skipped_execute_once,
                            "execute_once_reference_seed": first_outer_seed,
                            "seed_authority.mode_counts": campaign_profile["mode_counts"],
                            "seed_authority.evidence_class_counts": campaign_profile[
                                "evidence_class_counts"
                            ],
                            "all_complete": True,
                            "scientific_promotion": False,
                        },
                        "receipt_sha256",
                    )
                ],
                authorities=[
                    "scripts/generation1_cognitive_corpus.py",
                    "src/mop/studies/generation1_cognitive_corpus.py",
                    "configs/experiment/generation1_cognitive_corpus.json",
                    "scripts/generation1_resource_canary.py",
                    "src/mop/studies/generation1_resource_canary.py",
                    resource_canary["path"],
                ],
            )
        )
        seed_dependency = capsule_id

    corpus_output = "proof/GENERATION1_COGNITIVE_CORPUS.json"
    corpus_verification = "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
    capsules.append(
        _capsule(
            capsule_id="g1_cognitive_corpus_aggregate",
            kind="aggregate",
            priority=200,
            depends_on=seed_capsules,
            command=_python(
                "scripts/generation1_cognitive_corpus.py",
                "aggregate",
                "--config",
                "configs/experiment/generation1_cognitive_corpus.json",
                "--run-root",
                "runs/generation1/cognitive_corpus",
                "--out",
                corpus_output,
            ),
            resources=quick_resources("generation1_cognitive_corpus.py", wall=120),
            artifacts=[
                _artifact(
                    corpus_output,
                    "mop-generation1-cognitive-corpus/v2",
                    {
                        "campaign_id": EXPECTED_CAMPAIGN_ID,
                        "config.sha256": sha256_file(CORPUS_CONFIG),
                        "seed_count": len(corpus_config["seeds"]),
                        "eligible_experiment_count": experiment_count,
                        "complete_experiment_count": experiment_count,
                        "seed_authority_summary.mode_counts": campaign_profile["mode_counts"],
                        "seed_authority_summary.evidence_class_counts": campaign_profile[
                            "evidence_class_counts"
                        ],
                        "seed_authority_summary.no_pseudoreplication": True,
                        "operational_summary.invalid_attempt_receipt_count": 0,
                        "corpus_complete": True,
                        "scientific_promotion": False,
                    },
                    "corpus_sha256",
                )
            ],
            authorities=[
                "scripts/generation1_cognitive_corpus.py",
                "src/mop/studies/generation1_cognitive_corpus.py",
                "configs/experiment/generation1_cognitive_corpus.json",
            ],
        )
    )
    capsules.append(
        _capsule(
            capsule_id="g1_cognitive_corpus_verify",
            kind="verifier",
            priority=201,
            depends_on=["g1_cognitive_corpus_aggregate"],
            command=_python(
                "scripts/verify_generation1_cognitive_corpus.py",
                "--corpus",
                corpus_output,
                "--config",
                "configs/experiment/generation1_cognitive_corpus.json",
                "--run-root",
                "runs/generation1/cognitive_corpus",
                "--out",
                corpus_verification,
            ),
            resources=quick_resources("verify_generation1_cognitive_corpus.py", wall=120),
            artifacts=[
                _artifact(
                    corpus_verification,
                    "mop-generation1-cognitive-corpus-verification/v2",
                    {
                        "verification_complete": True,
                        "problems": [],
                        "config.sha256": sha256_file(CORPUS_CONFIG),
                        "checks.corpus_complete": True,
                        "checks.full_regeneration_match": True,
                        "checks.all_seed_receipts_valid": True,
                        "checks.all_attempt_receipts_valid": True,
                        "checks.all_cell_authorities_valid": True,
                        "checks.seed_authority_exact": True,
                        "checks.no_pseudoreplication": True,
                        "checks.independent_summary_match": True,
                        "checks.directional_inference_fail_closed": True,
                        "checks.all_mutations_rejected": True,
                        "authority_audit.expected_effective_cell_count": (
                            campaign_profile["effective_executions"]
                        ),
                        "authority_audit.selected_effective_cell_count": (
                            campaign_profile["effective_executions"]
                        ),
                        "scientific_promotion": False,
                    },
                    "verification_sha256",
                )
            ],
            authorities=[
                "scripts/verify_generation1_cognitive_corpus.py",
                "src/mop/studies/generation1_cognitive_corpus.py",
                "src/mop/studies/generation1_cognitive_corpus_verify.py",
                "configs/experiment/generation1_cognitive_corpus.json",
            ],
        )
    )
    capsules.append(
        _capsule(
            capsule_id="g1_empirical_report",
            kind="aggregate",
            priority=202,
            depends_on=["g1_cognitive_corpus_verify"],
            command=_python(
                "scripts/build_generation1_report.py",
                "--corpus",
                corpus_output,
                "--verification",
                corpus_verification,
                "--out",
                "proof/GENERATION1_EMPIRICAL_REPORT.json",
                "--text-out",
                "runs/generation1/GENERATION1_EMPIRICAL_REPORT.txt",
            ),
            resources=quick_resources("build_generation1_report.py", wall=120),
            artifacts=[
                _artifact(
                    "proof/GENERATION1_EMPIRICAL_REPORT.json",
                    "mop-generation1-empirical-report/v2",
                    {
                        "corpus.corpus_complete": True,
                        **{
                            f"corpus.verification_checks.{check}": True
                            for check in ADVANCED_VERIFIER_CHECKS
                        },
                        "resource_authority.source.path": resource_canary["path"],
                        "resource_authority.source.sha256": resource_canary["file_sha256"],
                        "resource_authority.recommendation.eligible": True,
                        "resource_authority.recommendation.recommended_max_workers": (
                            resource_canary["max_workers"]
                        ),
                        "resource_authority.recommendation."
                        "recommended_estimated_unified_memory_gb": resource_canary["memory_gb"],
                        "resource_authority.scientific_promotion": False,
                        "next_authority.ready_to_preregister_mechanism_epoch": True,
                        "scientific_promotion": False,
                    },
                    "report_sha256",
                )
            ],
            authorities=[
                "scripts/build_generation1_report.py",
                "src/mop/studies/generation1_report.py",
                "proof/ESCS_X0_EVENT_FORMATION.verification.json",
                "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
                "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
                "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
                resource_canary["path"],
            ],
        )
    )

    authority_rows = {_relative(path): _authority(path) for path in _iter_authority_files()}
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": PROGRAM_ID,
        "program_root": PROGRAM_ROOT,
        "policy": _authority("configs/local_execution_throttle.yaml"),
        "authorities": [authority_rows[key] for key in sorted(authority_rows)],
        "injection": {
            "inbox": f"{PROGRAM_ROOT}/control/inbox",
            "receipt_root": f"{PROGRAM_ROOT}/control/injection_receipts",
        },
        "control": {
            "throttle_state_root": "runs/local_throttle",
            "admission_samples": 3,
            "admission_interval_seconds": 15,
            "resource_retry_seconds": 120,
            "startup_ack_seconds": 120,
        },
        "capsules": capsules,
    }
    return {**core, "program_sha256": canonical_sha256(core)}


def _validate_runtime_compatibility(output: Path, expected_program_sha256: str) -> None:
    program = load_program(output)
    if program.program_sha256 != expected_program_sha256:
        raise ValueError("loaded Generation-1 program digest does not match the generated program")

    policy = load_policy(REPO_ROOT / program.policy.path)
    hard_wall_minutes = int(policy.limits["hard_wall_minutes"])
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    problems: list[str] = []
    for capsule in program.capsules:
        marker = capsule.resources.process_marker
        if marker not in known_markers:
            problems.append(
                f"{capsule.capsule_id}: process marker {marker!r} is absent from the live policy"
            )
        problems.extend(
            f"{capsule.capsule_id}: {problem}"
            for problem in capsule.task_declaration().validate(hard_wall_minutes)
        )
    if problems:
        raise ValueError("Generation-1 program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    program = build_program()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("Generation-1 program manifest must be inside the repository")
    if arguments.check:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != program:
            raise SystemExit("Generation-1 program manifest is stale")
    else:
        atomic_write_json(output, program)
    _validate_runtime_compatibility(output, program["program_sha256"])
    print(
        json.dumps(
            {
                "path": str(output),
                "program_id": program["program_id"],
                "program_sha256": program["program_sha256"],
                "capsule_count": len(program["capsules"]),
                "authority_count": len(program["authorities"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
