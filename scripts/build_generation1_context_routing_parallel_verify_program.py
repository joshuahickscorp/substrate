#!/usr/bin/env python3
"""Build the restart-safe C2 continuation with adaptive parallel verification."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_supervisor import (
    PROGRAM_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_program,
)
from mop.studio.local_throttle import load_policy

base: Any = import_module("build_generation1_context_routing_program")

PROGRAM_ID = "generation1-c2-context-routing-v1-adaptive25-parallel-verify"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_context_routing_adaptive25_parallel_verify.json"
VERIFY_CAPSULE_ID = "g1_c2_context_routing_verify_parallel"


def build_program(
    *,
    program_id: str = PROGRAM_ID,
    program_root: str | None = None,
) -> dict[str, Any]:
    resolved_root = program_root or f"runs/generation1/{program_id}"
    serial = base.build_program(program_id=program_id, program_root=resolved_root)
    capsules = list(serial["capsules"][:-1])
    capsules.append(
        base._capsule(
            capsule_id=VERIFY_CAPSULE_ID,
            kind="verifier",
            priority=505,
            depends_on=["g1_c2_context_routing_aggregate"],
            command=[
                ".venv/bin/python",
                "scripts/generation1_context_routing_parallel/verify.py",
                "--config",
                base.CONFIG_PATH,
                "--result",
                base.RESULT_PATH,
                "--out",
                base.VERIFICATION_PATH,
                "--idle-workers",
                "25",
                "--hawking-workers",
                "6",
            ],
            process_marker="verify.py",
            wall_minutes=300,
            artifact={
                "path": base.VERIFICATION_PATH,
                "schema": "mop-generation1-context-routing-verification/v1",
                "seal_field": "verification_sha256",
                "fields": {
                    "campaign_id": "generation1-c2-context-routing-v1",
                    "dataset_reproduction.expected_cells": base.SEED_COUNT * 5,
                    "dataset_reproduction.reproduced_cells": base.SEED_COUNT * 5,
                    "dataset_reproduction.all_dataset_and_metric_reproductions_passed": True,
                    "parallel_execution.idle_workers": 25,
                    "parallel_execution.hawking_workers": 6,
                    "parallel_execution.worker_result_count": base.SEED_COUNT * 5,
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
                "scripts/generation1_context_routing_parallel/verify.py",
                "src/mop/studies/generation1_context_routing_verify_parallel.py",
                "src/mop/studies/generation1_context_routing_verify.py",
                "src/mop/__init__.py",
                "src/mop/process_labels.py",
                "src/mop/studies/generation1_context_routing.py",
                "scripts/pr1_mode_error_disjointness.py",
                base.CONFIG_PATH,
                "proof/GENERATION1_COMPETENCE_ATLAS.json",
                "proof/GENERATION1_COMPETENCE_ATLAS.verification.json",
            ],
            resource_basis=(
                "independent regeneration of all 40,960 datasets and raw-prediction metrics in "
                "canonical coordinate order; the sealed Hawking queue selects 25 one-thread "
                "workers while idle or six while active; one fresh all-actor canary and eight "
                "adversarial mutations remain parent-recomputed under a 16-GiB process-tree cap"
            ),
            cpu_cores=25,
            adaptive=True,
        )
    )
    authority_paths = sorted(
        {
            "scripts/build_generation1_context_routing_program.py",
            "scripts/build_generation1_context_routing_parallel_verify_program.py",
            *(str(row["path"]) for capsule in capsules for row in capsule["authorities"]),
        }
    )
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": program_id,
        "program_root": resolved_root,
        "policy": serial["policy"],
        "authorities": [base._authority(path) for path in authority_paths],
        "injection": {
            "inbox": f"{PROGRAM_ROOT}/control/inbox",
            "receipt_root": f"{PROGRAM_ROOT}/control/injection_receipts",
        },
        "control": serial["control"],
        "capsules": capsules,
    }
    return {**core, "program_sha256": canonical_sha256(core)}


def _validate_runtime(output: Path, expected_sha256: str) -> None:
    program = load_program(output)
    if program.program_sha256 != expected_sha256:
        raise ValueError("loaded parallel-verifier program digest differs from generated digest")
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
        raise ValueError("parallel-verifier program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--program-id", default=PROGRAM_ID)
    parser.add_argument("--program-root")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    program = build_program(program_id=arguments.program_id, program_root=arguments.program_root)
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("parallel-verifier manifest must remain inside the repository")
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("parallel-verifier manifest is stale")
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
