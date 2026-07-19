#!/usr/bin/env python3
"""Build the adaptive, restart-safe Generation 1 C2 routing program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_supervisor import (
    CAPSULE_SCHEMA,
    PROGRAM_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
)
from mop.studio.local_throttle import (
    TASKPOLICY_ADAPTIVE_CAP_GB,
    TASKPOLICY_ADAPTIVE_PREFIX,
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
    load_policy,
)

PROGRAM_ID = "generation1-c2-context-routing-v1-adaptive25-labeled"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_context_routing_adaptive25_labeled.json"
POLICY_PATH = "configs/local_execution_throttle_v5_opportunistic.yaml"
CONFIG_PATH = "configs/experiment/generation1_context_routing.json"
WORK_ROOT = "runs/generation1/generation1-c2-context-routing-v1-adaptive25/context_routing_work"
RESULT_PATH = "proof/GENERATION1_CONTEXT_ROUTING.json"
VERIFICATION_PATH = "proof/GENERATION1_CONTEXT_ROUTING.verification.json"
SHARD_COUNT = 4
SEED_COUNT = 8192
CELLS_PER_SHARD = SEED_COUNT * 5 // SHARD_COUNT


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"C2 authority must be a regular file: {source}")
    return {"path": path, "sha256": sha256_file(source)}


def _capsule(
    *,
    capsule_id: str,
    kind: str,
    priority: int,
    depends_on: list[str],
    command: list[str],
    process_marker: str,
    wall_minutes: int,
    artifact: dict[str, Any],
    authority_paths: list[str],
    resource_basis: str,
    cpu_cores: int,
    adaptive: bool,
) -> dict[str, Any]:
    prefix = TASKPOLICY_ADAPTIVE_PREFIX if adaptive else TASKPOLICY_COEXISTENCE_PREFIX
    memory = TASKPOLICY_ADAPTIVE_CAP_GB if adaptive else TASKPOLICY_COEXISTENCE_CAP_GB
    core = {
        "schema": CAPSULE_SCHEMA,
        "id": capsule_id,
        "kind": kind,
        "priority": priority,
        "depends_on": depends_on,
        "command": [*prefix, *command],
        "cwd": ".",
        "environment": {
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "resources": {
            "lane": "cpu",
            "accelerator": "none",
            "cpu_cores": cpu_cores,
            "estimated_unified_memory_gb": memory,
            "estimated_mps_gb": 0.0,
            "resource_basis": resource_basis,
            "forecast_write_gb": 0.5 if adaptive else 0.1,
            "atomic_write_gb": 0.02,
            "wall_minutes": wall_minutes,
            "process_marker": process_marker,
        },
        "artifacts": [artifact],
        "authorities": [_authority(path) for path in authority_paths],
    }
    return {**core, "capsule_sha256": canonical_sha256(core)}


def build_program(
    *,
    program_id: str = PROGRAM_ID,
    program_root: str = PROGRAM_ROOT,
    work_root: str = WORK_ROOT,
) -> dict[str, Any]:
    common_authorities = [
        "src/mop/__init__.py",
        "src/mop/process_labels.py",
        "src/mop/studies/generation1_context_routing.py",
        "scripts/pr1_mode_error_disjointness.py",
        CONFIG_PATH,
        "proof/GENERATION1_COMPETENCE_ATLAS.json",
        "proof/GENERATION1_COMPETENCE_ATLAS.verification.json",
    ]
    capsules: list[dict[str, Any]] = []
    prior: list[str] = []
    for shard_index in range(SHARD_COUNT):
        capsule_id = f"g1_c2_context_routing_shard_{shard_index:02d}"
        shard_path = f"proof/GENERATION1_CONTEXT_ROUTING.shard_{shard_index:02d}.json"
        capsule = _capsule(
            capsule_id=capsule_id,
            kind="corpus",
            priority=500 + shard_index,
            depends_on=prior,
            command=[
                ".venv/bin/python",
                "scripts/generation1_context_routing/run_shard.py",
                "--config",
                CONFIG_PATH,
                "--work-root",
                work_root,
                "--out",
                shard_path,
                "--shard-index",
                str(shard_index),
                "--idle-workers",
                "25",
                "--hawking-workers",
                "6",
            ],
            process_marker="run_shard.py",
            wall_minutes=300,
            artifact={
                "path": shard_path,
                "schema": "mop-generation1-context-routing-shard/v1",
                "seal_field": "shard_sha256",
                "fields": {
                    "campaign_id": "generation1-c2-context-routing-v1",
                    "shard_index": shard_index,
                    "shard_count": SHARD_COUNT,
                    "grid.expected_seed_count": SEED_COUNT // SHARD_COUNT,
                    "grid.expected_cell_count": CELLS_PER_SHARD,
                    "grid.completed_cell_count": CELLS_PER_SHARD,
                    "complete": True,
                    "problems": [],
                    "activation_allowed": False,
                    "scientific_promotion": False,
                },
            },
            authority_paths=[
                "scripts/generation1_context_routing/run_shard.py",
                *common_authorities,
            ],
            resource_basis=(
                "10,240 parent-written atomic seed-difficulty cells in a continuously replenished "
                "pool; sealed Hawking queue state selects 25 one-thread workers while idle or six "
                "while active; nice priority 5, throttled disk I/O, and a 16-GiB process-tree cap "
                "bound the adaptive lane"
            ),
            cpu_cores=25,
            adaptive=True,
        )
        capsules.append(capsule)
        prior = [capsule_id]
    aggregate_id = "g1_c2_context_routing_aggregate"
    capsules.append(
        _capsule(
            capsule_id=aggregate_id,
            kind="aggregate",
            priority=504,
            depends_on=prior,
            command=[
                ".venv/bin/python",
                "scripts/generation1_context_routing/aggregate.py",
                "--config",
                CONFIG_PATH,
                "--work-root",
                work_root,
                "--out",
                RESULT_PATH,
            ],
            process_marker="aggregate.py",
            wall_minutes=120,
            artifact={
                "path": RESULT_PATH,
                "schema": "mop-generation1-context-routing/v1",
                "seal_field": "result_sha256",
                "fields": {
                    "campaign_id": "generation1-c2-context-routing-v1",
                    "grid.expected_seed_count": SEED_COUNT,
                    "grid.completed_seed_count": SEED_COUNT,
                    "grid.expected_cell_count": SEED_COUNT * 5,
                    "grid.completed_cell_count": SEED_COUNT * 5,
                    "decision.ready_to_train_dispatcher": False,
                    "complete": True,
                    "problems": [],
                    "activation_allowed": False,
                    "scientific_promotion": False,
                },
            },
            authority_paths=[
                "scripts/generation1_context_routing/aggregate.py",
                *common_authorities,
            ],
            resource_basis=(
                "streaming validation and aggregation of 40,960 sealed raw-prediction cells; "
                "one background CPU thread under a 4-GiB process cap"
            ),
            cpu_cores=1,
            adaptive=False,
        )
    )
    capsules.append(
        _capsule(
            capsule_id="g1_c2_context_routing_verify",
            kind="verifier",
            priority=505,
            depends_on=[aggregate_id],
            command=[
                ".venv/bin/python",
                "scripts/generation1_context_routing/verify.py",
                "--config",
                CONFIG_PATH,
                "--result",
                RESULT_PATH,
                "--out",
                VERIFICATION_PATH,
            ],
            process_marker="verify.py",
            wall_minutes=300,
            artifact={
                "path": VERIFICATION_PATH,
                "schema": "mop-generation1-context-routing-verification/v1",
                "seal_field": "verification_sha256",
                "fields": {
                    "campaign_id": "generation1-c2-context-routing-v1",
                    "dataset_reproduction.expected_cells": SEED_COUNT * 5,
                    "dataset_reproduction.reproduced_cells": SEED_COUNT * 5,
                    "dataset_reproduction.all_dataset_and_metric_reproductions_passed": True,
                    "fresh_actor_canary.passed": True,
                    "mutation_suite.count": 8,
                    "mutation_suite.rejected": 8,
                    "mutation_suite.all_rejected": True,
                    "verification_complete": True,
                    "problems": [],
                    "activation_allowed": False,
                    "scientific_promotion": False,
                },
            },
            authority_paths=[
                "scripts/generation1_context_routing/verify.py",
                "src/mop/studies/generation1_context_routing_verify.py",
                *common_authorities,
            ],
            resource_basis=(
                "independent regeneration of all 40,960 datasets and raw-prediction metrics, one "
                "fresh all-actor canary, and eight adversarial mutations under a 4-GiB cap"
            ),
            cpu_cores=1,
            adaptive=False,
        )
    )
    authority_paths = sorted(
        {
            "scripts/build_generation1_context_routing_program.py",
            *(str(row["path"]) for capsule in capsules for row in capsule["authorities"]),
        }
    )
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": program_id,
        "program_root": program_root,
        "policy": _authority(POLICY_PATH),
        "authorities": [_authority(path) for path in authority_paths],
        "injection": {
            "inbox": f"{program_root}/control/inbox",
            "receipt_root": f"{program_root}/control/injection_receipts",
        },
        "control": {
            "throttle_state_root": "runs/local_throttle",
            "admission_samples": 1,
            "admission_interval_seconds": 5,
            "resource_retry_seconds": 10,
            "startup_ack_seconds": 120,
        },
        "capsules": capsules,
    }
    return {**core, "program_sha256": canonical_sha256(core)}


def _validate_runtime(output: Path, expected_sha256: str) -> None:
    program = load_program(output)
    if program.program_sha256 != expected_sha256:
        raise ValueError("loaded C2 program digest differs from generated digest")
    policy = load_policy(REPO_ROOT / program.policy.path)
    hard_wall = int(policy.limits["hard_wall_minutes"])
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    problems: list[str] = []
    for capsule in program.capsules:
        if capsule.resources.process_marker not in known_markers:
            problems.append(f"{capsule.capsule_id}: unknown marker {capsule.resources.process_marker}")
        problems.extend(
            f"{capsule.capsule_id}: {problem}" for problem in capsule.task_declaration().validate(hard_wall)
        )
    if problems:
        raise ValueError("C2 program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    program = build_program()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("C2 program manifest must remain inside the repository")
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("Generation 1 C2 program manifest is stale")
    else:
        atomic_write_json(output, program)
    _validate_runtime(output, str(program["program_sha256"]))
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
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
