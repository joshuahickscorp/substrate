#!/usr/bin/env python3
"""Build the sealed 74-capsule Generation 1 successor-horizon program."""

# ruff: noqa: E402 - direct execution must bootstrap the repository before MOP imports

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
for _source_root in (REPO_BOOTSTRAP / "src", REPO_BOOTSTRAP):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mop.config import REPO_ROOT
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as horizon
from mop.studies import generation1_successor_horizon_verify as horizon_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics
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

PROGRAM_ID = horizon.PROGRAM_ID
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
DEFAULT_OUTPUT = horizon.PROGRAM_MANIFEST
POLICY_PATH = "configs/local_execution_throttle_v5_opportunistic.yaml"
CLI_PATH = "scripts/mop_generation1_successor_horizon.py"
BUILDER_PATH = "scripts/build_generation1_successor_horizon_program.py"
ADMISSION_PATH = f"{PROGRAM_ROOT}/admission.json"
RESULT_PATH = "proof/GENERATION1_SUCCESSOR_HORIZON.json"
VERIFICATION_PATH = "proof/GENERATION1_SUCCESSOR_HORIZON.verification.json"
REPORT_PATH = "runs/generation1/GENERATION1_SUCCESSOR_EVIDENCE_REPORT.md"
REPORT_RECEIPT_PATH = f"{PROGRAM_ROOT}/report_receipt.json"
CONSOLIDATED_RESULT_PATH = "proof/GENERATION1_CONSOLIDATED_FINAL_RESULT.json"
PROCESS_MARKER = Path(CLI_PATH).name
COMPUTE_WALL_MINUTES = 285


def _mechanism_sources() -> tuple[str, ...]:
    root = REPO_ROOT / "src/mop/mechanisms"
    return tuple(str(path.relative_to(REPO_ROOT)) for path in sorted(root.glob("*.py")))


CORE_RUNTIME_AUTHORITIES = (
    CLI_PATH,
    "src/mop/__init__.py",
    "src/mop/config.py",
    "src/mop/process_labels.py",
    "src/mop/studies/__init__.py",
    "src/mop/studies/generation1_successor_horizon.py",
    "src/mop/studies/generation1_successor_horizon_verify.py",
    "src/mop/studies/generation1_consolidated_final_campaign.py",
    "src/mop/studies/generation1_c3_d1_frozen_queue.py",
    "src/mop/studies/generation1_c3_router_redesign.py",
    "src/mop/studies/generation1_c3_dispatch.py",
    "src/mop/studies/generation1_c3_dispatch_queue.py",
    "src/mop/studies/generation1_context_routing.py",
    "src/mop/studies/generation1_successor_mechanics_queue.py",
    "src/mop/ladder/__init__.py",
    "src/mop/ladder/ladder_contracts.py",
    "src/mop/ladder/stage_ladder.py",
    "src/mop/ladder/stage3_registry.py",
    "src/mop/substrate/__init__.py",
    "src/mop/substrate/events.py",
    "scripts/pr1_mode_error_disjointness.py",
    "proof/GENERATION1_CONTEXT_ROUTING.json",
    "proof/GENERATION1_CONTEXT_ROUTING.verification.json",
    "proof/GENERATION1_CONSOLIDATED_FINAL_MANIFEST.json",
)
COMPUTE_RUNTIME_AUTHORITIES = tuple(sorted({*CORE_RUNTIME_AUTHORITIES, *_mechanism_sources()}))
PROGRAM_RUNTIME_AUTHORITIES = tuple(
    sorted(
        {
            BUILDER_PATH,
            *COMPUTE_RUNTIME_AUTHORITIES,
            "src/mop/studio/generation1_supervisor.py",
            "src/mop/studio/local_throttle.py",
            "src/mop/studio/external_coexistence.py",
            "src/mop/studio/task_policy_authority.py",
        }
    )
)


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"successor-horizon authority must be a regular file: {source}")
    return {"path": path, "sha256": sha256_file(source)}


def _artifact(
    path: str,
    schema: str,
    seal_field: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": path,
        "schema": schema,
        "seal_field": seal_field,
        "fields": fields,
    }


def _capsule(
    *,
    capsule_id: str,
    kind: str,
    priority: int,
    depends_on: list[str],
    command: list[str],
    artifact: dict[str, Any],
    wall_minutes: int,
    compute: bool,
    resource_basis: str,
    forecast_write_gb: float,
    atomic_write_gb: float,
) -> dict[str, Any]:
    prefix = TASKPOLICY_ADAPTIVE_PREFIX if compute else TASKPOLICY_COEXISTENCE_PREFIX
    authorities = COMPUTE_RUNTIME_AUTHORITIES if compute else CORE_RUNTIME_AUTHORITIES
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
            "cpu_cores": horizon.IDLE_WORKERS if compute else 1,
            "estimated_unified_memory_gb": (
                TASKPOLICY_ADAPTIVE_CAP_GB if compute else TASKPOLICY_COEXISTENCE_CAP_GB
            ),
            "estimated_mps_gb": 0.0,
            "resource_basis": resource_basis,
            "forecast_write_gb": forecast_write_gb,
            "atomic_write_gb": atomic_write_gb,
            "wall_minutes": wall_minutes,
            "process_marker": PROCESS_MARKER,
        },
        "artifacts": [artifact],
        "authorities": [_authority(path) for path in authorities],
    }
    return {**core, "capsule_sha256": canonical_sha256(core)}


def _python_command(command: str, *arguments: str) -> list[str]:
    return [".venv/bin/python", CLI_PATH, command, *arguments]


def _shard_path(epoch_index: int, lane: str, shard_index: int) -> str:
    epoch = horizon.EPOCH_IDS[epoch_index].lower()
    return f"{PROGRAM_ROOT}/shards/{epoch}/{lane}_{shard_index:02d}.json"


def _classification_path(epoch_index: int) -> str:
    return f"{PROGRAM_ROOT}/classifications/{horizon.EPOCH_IDS[epoch_index].lower()}.json"


def _admission_capsule() -> dict[str, Any]:
    return _capsule(
        capsule_id="g1_horizon_admit",
        kind="aggregate",
        priority=700,
        depends_on=[],
        command=_python_command(
            "admit",
            "--output",
            ADMISSION_PATH,
            "--final",
            CONSOLIDATED_RESULT_PATH,
        ),
        artifact=_artifact(
            ADMISSION_PATH,
            horizon.ADMISSION_SCHEMA,
            "admission_sha256",
            {
                "program_id": PROGRAM_ID,
                "epoch_ids": list(horizon.EPOCH_IDS),
                "fresh_cycle_indices": list(horizon.EPOCH_CYCLES),
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=20,
        compute=False,
        resource_basis=(
            "single-thread validation and sealing of the completed consolidated-v1 authority; "
            "no experiment execution and a kernel-enforced 4096-MiB taskpolicy cap"
        ),
        forecast_write_gb=0.02,
        atomic_write_gb=0.01,
    )


def _shard_capsule(*, epoch_index: int, lane: str, shard_index: int, dependency: str) -> dict[str, Any]:
    epoch = horizon.EPOCH_IDS[epoch_index].lower()
    capsule_id = f"g1_{epoch}_{lane}_shard_{shard_index:02d}"
    artifact_path = _shard_path(epoch_index, lane, shard_index)
    planned_minutes = (
        len(horizon.D1_PARTITIONS[shard_index]) * consolidated.D1_PLANNED_RUNG_SECONDS / 60.0
        if lane == "d1"
        else sum(
            mechanics.WORK_ITEMS[index].seed_count
            * mechanics.PLANNED_SECONDS_PER_SEED[mechanics.WORK_ITEMS[index].mechanism]
            for index in horizon.MECHANICS_PARTITIONS[shard_index]
        )
        / 60.0
    )
    return _capsule(
        capsule_id=capsule_id,
        kind="corpus",
        priority=701 + epoch_index * 20,
        depends_on=[dependency],
        command=_python_command(
            "run-shard",
            "--root",
            PROGRAM_ROOT,
            "--admission",
            ADMISSION_PATH,
            "--epoch-index",
            str(epoch_index),
            "--lane",
            lane,
            "--shard-index",
            str(shard_index),
            "--idle-workers",
            str(horizon.IDLE_WORKERS),
            "--hawking-workers",
            str(horizon.HAWKING_WORKERS),
            "--retry-limit",
            str(horizon.RETRY_LIMIT),
        ),
        artifact=_artifact(
            artifact_path,
            horizon.SHARD_SCHEMA,
            "shard_sha256",
            {
                "program_id": PROGRAM_ID,
                "epoch_id": horizon.EPOCH_IDS[epoch_index],
                "cycle_index": horizon.EPOCH_CYCLES[epoch_index],
                "lane": lane,
                "shard_index": shard_index,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=COMPUTE_WALL_MINUTES,
        compute=True,
        resource_basis=(
            f"checkpointed {lane} robustness shard planned at {planned_minutes:.1f} serial minutes; "
            "a dynamic process pool floats from one to twenty workers with live host load, under "
            "the adaptive 16384-MiB taskpolicy process-tree cap"
        ),
        forecast_write_gb=6.0 if lane == "d1" else 2.0,
        atomic_write_gb=0.5,
    )


def _classification_capsule(epoch_index: int, dependencies: list[str]) -> dict[str, Any]:
    epoch = horizon.EPOCH_IDS[epoch_index].lower()
    return _capsule(
        capsule_id=f"g1_{epoch}_classify",
        kind="aggregate",
        priority=702 + epoch_index * 20,
        depends_on=dependencies,
        command=_python_command(
            "classify",
            "--root",
            PROGRAM_ROOT,
            "--admission",
            ADMISSION_PATH,
            "--epoch-index",
            str(epoch_index),
        ),
        artifact=_artifact(
            _classification_path(epoch_index),
            horizon.CLASSIFICATION_SCHEMA,
            "classification_sha256",
            {
                "program_id": PROGRAM_ID,
                "epoch_id": horizon.EPOCH_IDS[epoch_index],
                "epoch_index": epoch_index,
                "cycle_index": horizon.EPOCH_CYCLES[epoch_index],
                "routing.past_or_active_work_mutable": False,
                "routing.future_change_requires_new_sealed_child": True,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=90,
        compute=False,
        resource_basis=(
            "single-thread bounded reconstruction of one epoch classification and next-epoch "
            "routing barrier under the coexistence taskpolicy cap"
        ),
        forecast_write_gb=0.1,
        atomic_write_gb=0.02,
    )


def _tail_capsules(last_classification: str) -> list[dict[str, Any]]:
    aggregate_id = "g1_horizon_aggregate"
    verify_id = "g1_horizon_verify"
    aggregate = _capsule(
        capsule_id=aggregate_id,
        kind="aggregate",
        priority=900,
        depends_on=[last_classification],
        command=_python_command(
            "aggregate",
            "--root",
            PROGRAM_ROOT,
            "--admission",
            ADMISSION_PATH,
            "--output",
            RESULT_PATH,
        ),
        artifact=_artifact(
            RESULT_PATH,
            horizon.RESULT_SCHEMA,
            "result_sha256",
            {
                "program_id": PROGRAM_ID,
                "grid.epoch_count": len(horizon.EPOCH_IDS),
                "grid.d1_shard_count": len(horizon.EPOCH_IDS) * horizon.D1_SHARD_COUNT,
                "grid.mechanics_shard_count": (len(horizon.EPOCH_IDS) * horizon.MECHANICS_SHARD_COUNT),
                "decision.bounded_horizon_complete": True,
                "decision.independent_artifact_verification_pending": True,
                "decision.independent_scientific_confirmation": False,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        ),
        wall_minutes=120,
        compute=False,
        resource_basis=(
            "single-thread binding of 65 shard receipts and five classifications into one sealed "
            "cross-epoch result under the coexistence taskpolicy cap"
        ),
        forecast_write_gb=0.2,
        atomic_write_gb=0.05,
    )
    verify = _capsule(
        capsule_id=verify_id,
        kind="verifier",
        priority=901,
        depends_on=[aggregate_id],
        command=_python_command(
            "verify",
            "--result",
            RESULT_PATH,
            "--output",
            VERIFICATION_PATH,
        ),
        artifact=_artifact(
            VERIFICATION_PATH,
            horizon_verify.VERIFICATION_SCHEMA,
            "verification_sha256",
            {
                "program_id": PROGRAM_ID,
                "checks.result_seal_valid": True,
                "checks.admission_and_consolidated_authority_valid": True,
                "checks.all_shards_and_raw_artifacts_valid": True,
                "checks.classifications_independently_reproduced": True,
                "checks.all_seed_intervals_disjoint": True,
                "checks.mutation_suite_passed": True,
                "checks.independent_generator_family_present": False,
                "recomputation.bound_shard_count": 65,
                "mutation_suite.count": 9,
                "mutation_suite.rejected": 9,
                "mutation_suite.all_rejected": True,
                "verification_complete": True,
                "independent_scientific_confirmation": False,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        ),
        wall_minutes=240,
        compute=False,
        resource_basis=(
            "separately authored one-thread streaming reconstruction of all raw receipts, seed "
            "intervals, classifications, aggregation bindings, and nine semantic mutations"
        ),
        forecast_write_gb=0.2,
        atomic_write_gb=0.05,
    )
    report = _capsule(
        capsule_id="g1_horizon_report",
        kind="aggregate",
        priority=902,
        depends_on=[verify_id],
        command=_python_command(
            "report",
            "--result",
            RESULT_PATH,
            "--verification",
            VERIFICATION_PATH,
            "--report",
            REPORT_PATH,
            "--receipt",
            REPORT_RECEIPT_PATH,
        ),
        artifact=_artifact(
            REPORT_RECEIPT_PATH,
            horizon.REPORT_RECEIPT_SCHEMA,
            "receipt_sha256",
            {
                "program_id": PROGRAM_ID,
                "result.path": RESULT_PATH,
                "verification.path": VERIFICATION_PATH,
                "report.path": REPORT_PATH,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        ),
        wall_minutes=20,
        compute=False,
        resource_basis=(
            "single-thread rendering of a non-authoritative Markdown view plus a sealed JSON "
            "receipt that binds the result, verification, and report bytes"
        ),
        forecast_write_gb=0.02,
        atomic_write_gb=0.01,
    )
    return [aggregate, verify, report]


def build_program() -> dict[str, Any]:
    capsules = [_admission_capsule()]
    previous = "g1_horizon_admit"
    for epoch_index in range(len(horizon.EPOCH_IDS)):
        shard_ids: list[str] = []
        for lane, count in (
            ("d1", horizon.D1_SHARD_COUNT),
            ("mechanics", horizon.MECHANICS_SHARD_COUNT),
        ):
            for shard_index in range(count):
                capsule = _shard_capsule(
                    epoch_index=epoch_index,
                    lane=lane,
                    shard_index=shard_index,
                    dependency=previous,
                )
                capsules.append(capsule)
                shard_ids.append(str(capsule["id"]))
        classification = _classification_capsule(epoch_index, shard_ids)
        capsules.append(classification)
        previous = str(classification["id"])
    capsules.extend(_tail_capsules(previous))
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": PROGRAM_ID,
        "program_root": PROGRAM_ROOT,
        "policy": _authority(POLICY_PATH),
        "authorities": [_authority(path) for path in PROGRAM_RUNTIME_AUTHORITIES],
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


def validate_runtime(path: Path, expected_sha256: str) -> None:
    program = load_program(path)
    if program.program_sha256 != expected_sha256:
        raise ValueError("loaded successor-horizon digest differs from generated digest")
    policy = load_policy(REPO_ROOT / program.policy.path)
    hard_wall = int(policy.limits["hard_wall_minutes"])
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    problems: list[str] = []
    for capsule in program.capsules:
        if capsule.resources.wall_minutes >= 300:
            problems.append(f"{capsule.capsule_id}: wall must remain below 300 minutes")
        if capsule.resources.process_marker not in known_markers:
            problems.append(f"{capsule.capsule_id}: unknown marker {capsule.resources.process_marker}")
        problems.extend(
            f"{capsule.capsule_id}: {problem}" for problem in capsule.task_declaration().validate(hard_wall)
        )
    if problems:
        raise ValueError("successor-horizon program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("successor-horizon manifest must remain inside the repository")
    program = build_program()
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("successor-horizon manifest is stale")
    else:
        atomic_write_json(output, program)
    validate_runtime(output, str(program["program_sha256"]))
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
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
