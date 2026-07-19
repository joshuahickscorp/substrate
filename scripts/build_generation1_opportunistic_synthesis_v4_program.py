#!/usr/bin/env python3
"""Build the background-QoS post-recovery Generation 1 synthesis program."""

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

DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_evidence_synthesis_recovery_v4_opportunistic.json"
PROGRAM_ID = "generation1-evidence-synthesis-recovery4-opportunistic"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
RECOVERY_PROGRAM_ID = "generation1-empirical-cognitive-corpus-v2-recovery4-opportunistic"
RECOVERY_STATE = f"runs/generation1/{RECOVERY_PROGRAM_ID}/program_state.json"
POLICY_PATH = "configs/local_execution_throttle_v5_opportunistic.yaml"


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"synthesis authority must be a regular file: {source}")
    return {"path": path, "sha256": sha256_file(source)}


def _capsule(
    *,
    capsule_id: str,
    kind: str,
    priority: int,
    depends_on: list[str],
    command: list[str],
    marker: str,
    artifact: dict[str, Any],
    authority_paths: list[str],
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
            "cpu_cores": 1,
            "estimated_unified_memory_gb": TASKPOLICY_COEXISTENCE_CAP_GB,
            "estimated_mps_gb": 0.0,
            "resource_basis": (
                "read-only deterministic aggregation over the verified Generation 1 corpus; "
                "one CPU thread, bounded outputs, and a separately authored verifier"
            ),
            "forecast_write_gb": 0.25,
            "atomic_write_gb": 0.1,
            "wall_minutes": 120,
            "process_marker": marker,
        },
        "artifacts": [artifact],
        "authorities": [_authority(path) for path in authority_paths],
    }
    return {**core, "capsule_sha256": canonical_sha256(core)}


def build_program() -> dict[str, Any]:
    synthesis_id = "g1_evidence_synthesis"
    synthesis = _capsule(
        capsule_id=synthesis_id,
        kind="aggregate",
        priority=300,
        depends_on=[],
        command=[
            ".venv/bin/python",
            "scripts/generation1_evidence_synthesis/build_generation1_report.py",
            "--corpus",
            "proof/GENERATION1_COGNITIVE_CORPUS.json",
            "--verification",
            "proof/GENERATION1_COGNITIVE_CORPUS.verification.json",
            "--report",
            "proof/GENERATION1_EMPIRICAL_REPORT.json",
            "--program-state",
            RECOVERY_STATE,
            "--out",
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "--text-out",
            "runs/generation1/GENERATION1_EVIDENCE_SYNTHESIS.txt",
        ],
        marker="build_generation1_report.py",
        artifact={
            "path": "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "schema": "mop-generation1-evidence-synthesis/v1",
            "seal_field": "synthesis_sha256",
            "fields": {
                "activation_allowed": False,
                "scientific_promotion": False,
                "base_runtime_accounting.program_id": RECOVERY_PROGRAM_ID,
                "base_runtime_accounting.base_capsule_count": 3,
                "base_runtime_accounting.status_counts.complete": 3,
            },
        },
        authority_paths=[
            "scripts/generation1_evidence_synthesis/build_generation1_report.py",
            "src/mop/studies/generation1_evidence_synthesis.py",
        ],
    )
    verification = _capsule(
        capsule_id="g1_evidence_synthesis_verify",
        kind="verifier",
        priority=301,
        depends_on=[synthesis_id],
        command=[
            ".venv/bin/python",
            "scripts/generation1_evidence_synthesis/verify_generation1_cognitive_corpus.py",
            "--corpus",
            "proof/GENERATION1_COGNITIVE_CORPUS.json",
            "--verification",
            "proof/GENERATION1_COGNITIVE_CORPUS.verification.json",
            "--report",
            "proof/GENERATION1_EMPIRICAL_REPORT.json",
            "--program-state",
            RECOVERY_STATE,
            "--synthesis",
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "--out",
            "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
        ],
        marker="verify_generation1_cognitive_corpus.py",
        artifact={
            "path": "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
            "schema": "mop-generation1-evidence-synthesis-verification/v1",
            "seal_field": "verification_sha256",
            "fields": {
                "verification_complete": True,
                "problems": [],
                "mutation_suite.count": 8,
                "mutation_suite.rejected": 8,
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        },
        authority_paths=[
            "scripts/generation1_evidence_synthesis/verify_generation1_cognitive_corpus.py",
            "src/mop/studies/generation1_evidence_synthesis_verify.py",
        ],
    )
    capsules = [synthesis, verification]
    authority_paths = sorted(
        {
            "scripts/build_generation1_opportunistic_synthesis_v4_program.py",
            *(str(row["path"]) for capsule in capsules for row in capsule["authorities"]),
        }
    )
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": PROGRAM_ID,
        "program_root": PROGRAM_ROOT,
        "policy": _authority(POLICY_PATH),
        "authorities": [_authority(path) for path in authority_paths],
        "injection": {
            "inbox": f"{PROGRAM_ROOT}/control/inbox",
            "receipt_root": f"{PROGRAM_ROOT}/control/injection_receipts",
        },
        "control": {
            "throttle_state_root": "runs/local_throttle",
            "admission_samples": 3,
            "admission_interval_seconds": 15,
            "resource_retry_seconds": 30,
            "startup_ack_seconds": 120,
        },
        "capsules": capsules,
    }
    return {**core, "program_sha256": canonical_sha256(core)}


def _validate_runtime(output: Path, expected_sha256: str) -> None:
    program = load_program(output)
    if program.program_sha256 != expected_sha256:
        raise ValueError("loaded synthesis program digest differs from generated digest")
    policy = load_policy(REPO_ROOT / program.policy.path)
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    hard_wall_minutes = int(policy.limits["hard_wall_minutes"])
    problems: list[str] = []
    for capsule in program.capsules:
        if capsule.resources.process_marker not in known_markers:
            problems.append(f"{capsule.capsule_id}: unknown marker {capsule.resources.process_marker}")
        problems.extend(
            f"{capsule.capsule_id}: {problem}"
            for problem in capsule.task_declaration().validate(hard_wall_minutes)
        )
    if problems:
        raise ValueError("synthesis program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("synthesis program manifest must remain inside the repository")
    program = build_program()
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("Generation 1 synthesis program manifest is stale")
    else:
        atomic_write_json(output, program)
    _validate_runtime(output, str(program["program_sha256"]))
    print(
        json.dumps(
            {
                "path": str(output),
                "program_id": PROGRAM_ID,
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
