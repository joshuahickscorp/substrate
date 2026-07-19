#!/usr/bin/env python3
"""Build the one-thread opportunistic Generation 1 C1 atlas program."""

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
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
    load_policy,
)

DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_competence_atlas_opportunistic.json"
PROGRAM_ID = "generation1-c1-competence-atlas-v1-opportunistic"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
POLICY_PATH = "configs/local_execution_throttle_v5_opportunistic.yaml"
CONFIG_PATH = "configs/experiment/generation1_competence_atlas.json"
ATLAS_PATH = "proof/GENERATION1_COMPETENCE_ATLAS.json"
VERIFICATION_PATH = "proof/GENERATION1_COMPETENCE_ATLAS.verification.json"
MAX_SEED_WORKERS = 6


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"atlas authority must be a regular file: {source}")
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
) -> dict[str, Any]:
    core = {
        "schema": CAPSULE_SCHEMA,
        "id": capsule_id,
        "kind": kind,
        "priority": priority,
        "depends_on": depends_on,
        "command": [*TASKPOLICY_COEXISTENCE_PREFIX, *command],
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
            "estimated_unified_memory_gb": TASKPOLICY_COEXISTENCE_CAP_GB,
            "estimated_mps_gb": 0.0,
            "resource_basis": resource_basis,
            "forecast_write_gb": 0.1,
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
    atlas_work_root: str | None = None,
    admission_samples: int = 3,
    admission_interval_seconds: int = 15,
    resource_retry_seconds: int = 30,
    seed_workers: int = 1,
) -> dict[str, Any]:
    work_root = atlas_work_root or f"{program_root}/atlas_work"
    run_id = "g1_c1_difficulty_atlas"
    run_capsule = _capsule(
        capsule_id=run_id,
        kind="corpus",
        priority=400,
        depends_on=[],
        command=[
            ".venv/bin/python",
            "scripts/generation1_competence_atlas/generation1_cognitive_corpus.py",
            "--config",
            CONFIG_PATH,
            "--work-root",
            work_root,
            "--out",
            ATLAS_PATH,
            "--seed-workers",
            str(seed_workers),
        ],
        process_marker="generation1_cognitive_corpus.py",
        wall_minutes=300,
        artifact={
            "path": ATLAS_PATH,
            "schema": "mop-generation1-competence-atlas/v1",
            "seal_field": "atlas_sha256",
            "fields": {
                "campaign_id": "generation1-c1-competence-atlas-v1",
                "grid.expected_seed_count": 48,
                "grid.completed_seed_count": 48,
                "grid.expected_seed_difficulty_cells": 240,
                "grid.completed_seed_difficulty_cells": 240,
                "complete": True,
                "problems": [],
                "decision.ready_to_train_dispatcher": False,
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        },
        authority_paths=[
            "scripts/generation1_competence_atlas/generation1_cognitive_corpus.py",
            "src/mop/studies/generation1_competence_atlas.py",
            "scripts/pr1_mode_error_disjointness.py",
            CONFIG_PATH,
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
        ],
        resource_basis=(
            "restart-safe 48-seed by five-difficulty generated competence atlas; bounded "
            f"{seed_workers}-process seed pool under background taskpolicy with parent-owned "
            "atomic per-seed receipts and a 4096-MiB process-tree envelope"
        ),
        cpu_cores=seed_workers,
    )
    verify_capsule = _capsule(
        capsule_id="g1_c1_difficulty_atlas_verify",
        kind="verifier",
        priority=401,
        depends_on=[run_id],
        command=[
            ".venv/bin/python",
            "scripts/generation1_competence_atlas/verify_generation1_cognitive_corpus.py",
            "--config",
            CONFIG_PATH,
            "--atlas",
            ATLAS_PATH,
            "--out",
            VERIFICATION_PATH,
        ],
        process_marker="verify_generation1_cognitive_corpus.py",
        wall_minutes=120,
        artifact={
            "path": VERIFICATION_PATH,
            "schema": "mop-generation1-competence-atlas-verification/v1",
            "seal_field": "verification_sha256",
            "fields": {
                "campaign_id": "generation1-c1-competence-atlas-v1",
                "verification_complete": True,
                "problems": [],
                "mutation_suite.count": 8,
                "mutation_suite.rejected": 8,
                "mutation_suite.all_rejected": True,
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        },
        authority_paths=[
            "scripts/generation1_competence_atlas/verify_generation1_cognitive_corpus.py",
            "src/mop/studies/generation1_competence_atlas_verify.py",
            "src/mop/studies/generation1_competence_atlas.py",
            "scripts/pr1_mode_error_disjointness.py",
            CONFIG_PATH,
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
        ],
        resource_basis=(
            "independent raw-prediction, generated-dataset, aggregate, canary, and mutation "
            "verification; one background CPU thread with a 4096-MiB process cap"
        ),
        cpu_cores=1,
    )
    capsules = [run_capsule, verify_capsule]
    authority_paths = sorted(
        {
            "scripts/build_generation1_competence_atlas_program.py",
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
            "admission_samples": admission_samples,
            "admission_interval_seconds": admission_interval_seconds,
            "resource_retry_seconds": resource_retry_seconds,
            "startup_ack_seconds": 120,
        },
        "capsules": capsules,
    }
    return {**core, "program_sha256": canonical_sha256(core)}


def _validate_runtime(output: Path, expected_sha256: str) -> None:
    program = load_program(output)
    if program.program_sha256 != expected_sha256:
        raise ValueError("loaded atlas program digest differs from generated digest")
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
        raise ValueError("atlas program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--program-id", default=PROGRAM_ID)
    parser.add_argument("--program-root")
    parser.add_argument("--atlas-work-root")
    parser.add_argument("--admission-samples", type=int, default=3)
    parser.add_argument("--admission-interval-seconds", type=int, default=15)
    parser.add_argument("--resource-retry-seconds", type=int, default=30)
    parser.add_argument(
        "--seed-workers", type=int, choices=range(1, MAX_SEED_WORKERS + 1), default=1
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not arguments.program_id.strip() or "/" in arguments.program_id:
        raise SystemExit("program id must be a nonempty path-free identifier")
    program_root = arguments.program_root or f"runs/generation1/{arguments.program_id}"
    atlas_work_root = arguments.atlas_work_root or f"{program_root}/atlas_work"
    for name in (
        "admission_samples",
        "admission_interval_seconds",
        "resource_retry_seconds",
    ):
        if getattr(arguments, name) <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("atlas program manifest must remain inside the repository")
    program = build_program(
        program_id=arguments.program_id,
        program_root=program_root,
        atlas_work_root=atlas_work_root,
        admission_samples=arguments.admission_samples,
        admission_interval_seconds=arguments.admission_interval_seconds,
        resource_retry_seconds=arguments.resource_retry_seconds,
        seed_workers=arguments.seed_workers,
    )
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("Generation 1 competence-atlas program manifest is stale")
    else:
        atomic_write_json(output, program)
    _validate_runtime(output, str(program["program_sha256"]))
    print(
        json.dumps(
            {
                "path": str(output),
                "program_id": arguments.program_id,
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
