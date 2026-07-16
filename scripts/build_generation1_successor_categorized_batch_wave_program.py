#!/usr/bin/env python3
"""Build the sealed 59-capsule categorized successor scaffold program."""

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
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studies import generation1_successor_categorized_batch_wave_verify as verifier
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

PROGRAM_ID = wave.PROGRAM_ID
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
DEFAULT_OUTPUT = wave.PROGRAM_MANIFEST
POLICY_PATH = "configs/local_execution_throttle_v5_opportunistic.yaml"
CLI_PATH = (
    "scripts/generation1_successor_categorized_batch_wave/mop_generation1_successor_categorized_batch_wave.py"
)
BUILDER_PATH = "scripts/build_generation1_successor_categorized_batch_wave_program.py"
RUNTIME_PATH = "src/mop/studies/generation1_successor_categorized_batch_wave.py"
VERIFIER_PATH = "src/mop/studies/generation1_successor_categorized_batch_wave_verify.py"
PROCESS_MARKER = "mop_generation1_successor_horizon.py"

RESULT_PATH = "proof/GENERATION1_SUCCESSOR_CATEGORIZED_BATCH_WAVE.json"
VERIFICATION_PATH = "proof/GENERATION1_SUCCESSOR_CATEGORIZED_BATCH_WAVE.verification.json"
REPORT_PATH = "runs/generation1/GENERATION1_SUCCESSOR_CATEGORIZED_BATCH_WAVE.md"
REPORT_RECEIPT_PATH = f"{PROGRAM_ROOT}/report_receipt.json"
PARENT_RESULT_PATH = "proof/GENERATION1_SUCCESSOR_HORIZON_V2.json"
PARENT_VERIFICATION_PATH = "proof/GENERATION1_SUCCESSOR_HORIZON_V2.verification.json"
PARENT_REPORT_RECEIPT_PATH = "runs/generation1/generation1-successor-horizon-v2/report_receipt.json"
D1_RUNG_ROOT_PATH = str(wave.DEFAULT_D1_RUNG_ROOT.relative_to(REPO_ROOT))

CORE_RUNTIME_AUTHORITIES = (
    CLI_PATH,
    RUNTIME_PATH,
    VERIFIER_PATH,
    "src/mop/__init__.py",
    "src/mop/config.py",
    "src/mop/studies/__init__.py",
    "src/mop/studies/generation1_successor_horizon.py",
    "src/mop/studies/generation1_successor_horizon_verify.py",
    "src/mop/studies/generation1_successor_horizon_v2.py",
    "src/mop/studies/generation1_successor_horizon_v2_verify.py",
    "src/mop/studies/generation1_d1_frozen_verify.py",
    "src/mop/studies/generation1_d1_redesign_v2.py",
    "src/mop/studies/generation1_consolidated_final_campaign.py",
    "src/mop/studies/generation1_successor_mechanics_queue.py",
    "configs/campaign/generation1_successor_horizon_v1.json",
    "configs/campaign/generation1_successor_horizon_v2.json",
    "scripts/generation1_successor_horizon_v2/mop_generation1_successor_horizon.py",
    "src/mop/studio/generation1_supervisor.py",
)
PROGRAM_RUNTIME_AUTHORITIES = tuple(
    sorted(
        {
            BUILDER_PATH,
            *CORE_RUNTIME_AUTHORITIES,
            "src/mop/studio/local_throttle.py",
            "src/mop/studio/external_coexistence.py",
            "src/mop/studio/task_policy_authority.py",
        }
    )
)


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"categorized-wave authority must be a regular file: {source}")
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
    resource_basis: str,
    compute: bool = False,
    forecast_write_gb: float = 0.02,
    atomic_write_gb: float = 0.01,
) -> dict[str, Any]:
    prefix = TASKPOLICY_ADAPTIVE_PREFIX if compute else TASKPOLICY_COEXISTENCE_PREFIX
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
            "cpu_cores": wave.IDLE_WORKERS if compute else 1,
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
        "authorities": [_authority(path) for path in CORE_RUNTIME_AUTHORITIES],
    }
    return {**core, "capsule_sha256": canonical_sha256(core)}


def _python_command(command: str, *arguments: str) -> list[str]:
    return [
        f"MOP_REGISTERED_PROCESS_FAMILY={PROCESS_MARKER}",
        ".venv/bin/python",
        CLI_PATH,
        command,
        *arguments,
    ]


def _gate_path(gate_id: str) -> str:
    return f"{PROGRAM_ROOT}/gates/{gate_id}.json"


def _category_path(epoch_index: int, category_id: str) -> str:
    return f"{PROGRAM_ROOT}/waves/{wave.EPOCH_IDS[epoch_index].lower()}/{category_id}.json"


def _classification_path(epoch_index: int) -> str:
    return f"{PROGRAM_ROOT}/classifications/{wave.EPOCH_IDS[epoch_index].lower()}.json"


def _gate_capsule(gate_index: int, dependency: str | None) -> dict[str, Any]:
    gate_id = wave.GATE_IDS[gate_index]
    arguments = [
        "--root",
        PROGRAM_ROOT,
        "--gate-index",
        str(gate_index),
    ]
    if gate_index == 0:
        arguments.extend(
            (
                "--parent-result",
                PARENT_RESULT_PATH,
                "--parent-verification",
                PARENT_VERIFICATION_PATH,
                "--parent-report-receipt",
                PARENT_REPORT_RECEIPT_PATH,
            )
        )
    if gate_index == 1:
        arguments.extend(
            (
                "--d1-result",
                "proof/GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.json",
                "--d1-rung-root",
                D1_RUNG_ROOT_PATH,
            )
        )
    return _capsule(
        capsule_id=gate_id,
        kind="aggregate",
        priority=700 + gate_index,
        depends_on=[] if dependency is None else [dependency],
        command=_python_command("materialize-gate", *arguments),
        artifact=_artifact(
            _gate_path(gate_id),
            wave.GATE_SCHEMA,
            "gate_sha256",
            {
                "program_id": PROGRAM_ID,
                "gate_id": gate_id,
                "gate_index": gate_index,
                "compute_started": False,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=30 if gate_index == 0 else 15,
        resource_basis=(
            "single-thread immutable transition receipt; the first gate validates and binds "
            "the clean v2 manifest, zero-injection terminal status, result, verification, and "
            "report; the D1 verification gate replays and byte-binds every completed raw rung "
            "without restarting D1 compute, while later gates retire, screen, and freeze"
        ),
    )


def _category_capsule(
    epoch_index: int,
    category_id: str,
    dependency: str,
) -> dict[str, Any]:
    epoch_id = wave.EPOCH_IDS[epoch_index]
    capsule_id = f"{epoch_id.lower()}_{category_id}"
    planned = wave.category_planned_serial_seconds(epoch_index, category_id)
    return _capsule(
        capsule_id=capsule_id,
        kind="corpus",
        priority=720 + epoch_index * 10,
        depends_on=[dependency],
        command=_python_command(
            "run-category",
            "--root",
            PROGRAM_ROOT,
            "--epoch-index",
            str(epoch_index),
            "--category",
            category_id,
        ),
        artifact=_artifact(
            _category_path(epoch_index, category_id),
            wave.CATEGORY_SCHEMA,
            "category_sha256",
            {
                "program_id": PROGRAM_ID,
                "epoch_id": epoch_id,
                "cycle_index": wave.EPOCH_CYCLES[epoch_index],
                "category.id": category_id,
                "balanced_planning_shard_count": wave.INTERNAL_SHARD_COUNT,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=285,
        resource_basis=(
            f"fresh-cycle deterministic mechanics execution for "
            f"{wave.CATEGORY_NAMES[category_id]} in {epoch_id}; a dynamic process pool capped "
            "at eight workers executes retained lanes only, while eight balanced planning "
            f"shards describe {planned / 3_600:.2f} maximum serial hours and restart-safe "
            "raw receipts"
        ),
        compute=True,
        forecast_write_gb=4.0,
        atomic_write_gb=0.5,
    )


def _wave_classification_capsule(
    epoch_index: int,
    dependencies: list[str],
) -> dict[str, Any]:
    epoch_id = wave.EPOCH_IDS[epoch_index]
    return _capsule(
        capsule_id=f"{epoch_id.lower()}_classify",
        kind="aggregate",
        priority=721 + epoch_index * 10,
        depends_on=dependencies,
        command=_python_command(
            "classify-wave",
            "--root",
            PROGRAM_ROOT,
            "--epoch-index",
            str(epoch_index),
        ),
        artifact=_artifact(
            _classification_path(epoch_index),
            wave.CLASSIFICATION_SCHEMA,
            "classification_sha256",
            {
                "program_id": PROGRAM_ID,
                "epoch_id": epoch_id,
                "cycle_index": wave.EPOCH_CYCLES[epoch_index],
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=20,
        resource_basis=(
            f"single-thread serial barrier for {epoch_id}; binds all six category receipts "
            "and carries only null-safe, non-resurrecting routes into the next wave"
        ),
    )


def _integration_capsules(last_wave: str) -> list[dict[str, Any]]:
    integration_id = "i1_integrate"
    integration_path = f"{PROGRAM_ROOT}/integration/i1.json"
    classification_path = f"{PROGRAM_ROOT}/integration/i1_classification.json"
    integration = _capsule(
        capsule_id=integration_id,
        kind="aggregate",
        priority=800,
        depends_on=[last_wave],
        command=_python_command("run-integration", "--root", PROGRAM_ROOT),
        artifact=_artifact(
            integration_path,
            wave.INTEGRATION_SCHEMA,
            "integration_sha256",
            {
                "program_id": PROGRAM_ID,
                "integration_id": "I1",
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=180,
        resource_basis=(
            "conditional post-W07 G1-I1 evaluation; executes fresh mechanics through a dynamic "
            "process pool capped at eight workers only when initial v2 admission and every "
            "declared dependency lane remain clean through W07, otherwise seals a pruned receipt"
        ),
        compute=True,
        forecast_write_gb=2.0,
        atomic_write_gb=0.5,
    )
    classification = _capsule(
        capsule_id="i1_classify",
        kind="aggregate",
        priority=801,
        depends_on=[integration_id],
        command=_python_command("classify-integration", "--root", PROGRAM_ROOT),
        artifact=_artifact(
            classification_path,
            wave.INTEGRATION_CLASSIFICATION_SCHEMA,
            "classification_sha256",
            {
                "program_id": PROGRAM_ID,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
                "independent_scientific_confirmation": False,
            },
        ),
        wall_minutes=20,
        resource_basis=(
            "single-thread I1 classification that exposes the final route inventory for "
            "operator review and explicitly withholds future-compute authority"
        ),
    )
    return [integration, classification]


def _tail_capsules() -> list[dict[str, Any]]:
    aggregate = _capsule(
        capsule_id="categorized_wave_aggregate",
        kind="aggregate",
        priority=900,
        depends_on=["i1_classify"],
        command=_python_command(
            "aggregate",
            "--root",
            PROGRAM_ROOT,
            "--output",
            RESULT_PATH,
        ),
        artifact=_artifact(
            RESULT_PATH,
            wave.RESULT_SCHEMA,
            "result_sha256",
            {
                "program_id": PROGRAM_ID,
                "grid.manifest_capsule_count": wave.CAPSULE_COUNT,
                "grid.balanced_planning_shards_per_compute_capsule": (wave.INTERNAL_SHARD_COUNT),
                "grid.maximum_raw_receipt_count": wave.MAXIMUM_RAW_RECEIPT_COUNT,
                "execution.maximum_planned_serial_seconds": (wave.planned_program_compute_seconds()),
                "decision.categorized_mechanics_complete": True,
                "decision.d1_redesign_efficacy_executed": False,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        ),
        wall_minutes=45,
        resource_basis=(
            "single-thread binding of five transition gates, 42 executed-or-pruned category "
            "receipts, seven wave classifiers, I1, and the I1 classifier into one sealed result"
        ),
        forecast_write_gb=0.05,
        atomic_write_gb=0.02,
    )
    verify = _capsule(
        capsule_id="categorized_wave_verify",
        kind="verifier",
        priority=901,
        depends_on=["categorized_wave_aggregate"],
        command=_python_command(
            "verify",
            "--result",
            RESULT_PATH,
            "--output",
            VERIFICATION_PATH,
        ),
        artifact=_artifact(
            VERIFICATION_PATH,
            verifier.VERIFICATION_SCHEMA,
            "verification_sha256",
            {
                "program_id": PROGRAM_ID,
                "claim_scope": verifier.CLAIM_SCOPE,
                "checks.result_seal_valid": True,
                "checks.five_d1_transition_gates_valid": True,
                "checks.single_post_w07_i1_integration_valid": True,
                "checks.fresh_mechanics_receipts_valid": True,
                "checks.observed_execution_accounted": True,
                "checks.d1_redesign_efficacy_not_executed": True,
                "checks.mutation_suite_passed": True,
                "recomputation.manifest_capsule_count": wave.CAPSULE_COUNT,
                "mutation_suite.count": verifier.MUTATION_COUNT,
                "mutation_suite.rejected": verifier.MUTATION_COUNT,
                "mutation_suite.all_rejected": True,
                "verification_complete": True,
                "independent_scientific_confirmation": False,
                "complete": True,
                "problems": [],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
        ),
        wall_minutes=90,
        resource_basis=(
            "separately authored single-thread artifact replay and semantic mutation "
            "verification of the complete categorized mechanics program"
        ),
        forecast_write_gb=0.05,
        atomic_write_gb=0.02,
    )
    report = _capsule(
        capsule_id="categorized_wave_report",
        kind="aggregate",
        priority=902,
        depends_on=["categorized_wave_verify"],
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
            wave.REPORT_RECEIPT_SCHEMA,
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
        resource_basis=(
            "single-thread rendering of a non-authoritative Markdown view plus a sealed "
            "receipt binding the executable mechanics result, verification, and report bytes"
        ),
    )
    return [aggregate, verify, report]


def build_program() -> dict[str, Any]:
    capsules = []
    previous: str | None = None
    for gate_index, gate_id in enumerate(wave.GATE_IDS):
        capsules.append(_gate_capsule(gate_index, previous))
        previous = gate_id
    assert previous is not None
    for epoch_index in range(len(wave.EPOCH_IDS)):
        category_ids = []
        for category_id in wave.CATEGORY_IDS:
            capsule = _category_capsule(epoch_index, category_id, previous)
            capsules.append(capsule)
            category_ids.append(str(capsule["id"]))
        classification = _wave_classification_capsule(epoch_index, category_ids)
        capsules.append(classification)
        previous = str(classification["id"])
    capsules.extend(_integration_capsules(previous))
    capsules.extend(_tail_capsules())
    if len(capsules) != wave.CAPSULE_COUNT:
        raise ValueError(f"categorized-wave capsule count drifted: {len(capsules)}")
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
        raise ValueError("loaded categorized-wave digest differs from generated digest")
    policy = load_policy(REPO_ROOT / program.policy.path)
    hard_wall = int(policy.limits["hard_wall_minutes"])
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    problems: list[str] = []
    for capsule in program.capsules:
        if capsule.resources.wall_minutes >= 300:
            problems.append(f"{capsule.capsule_id}: wall must remain below 300 minutes")
        if capsule.resources.process_marker not in known_markers:
            problems.append(f"{capsule.capsule_id}: unknown marker")
        problems.extend(
            f"{capsule.capsule_id}: {problem}" for problem in capsule.task_declaration().validate(hard_wall)
        )
    if problems:
        raise ValueError("categorized-wave program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("categorized-wave manifest must remain inside the repository")
    program = build_program()
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("categorized-wave manifest is stale")
        if output.stat().st_mode & 0o777 != 0o644:
            raise SystemExit("categorized-wave manifest permissions drifted")
    else:
        atomic_write_json(output, program)
        output.chmod(0o644)
    validate_runtime(output, str(program["program_sha256"]))
    print(
        json.dumps(
            {
                "path": str(output),
                "program_id": PROGRAM_ID,
                "program_sha256": program["program_sha256"],
                "capsule_count": len(program["capsules"]),
                "authority_count": len(program["authorities"]),
                "maximum_planned_serial_hours": wave.planned_serial_hours(),
                "maximum_ideal_eight_worker_hours": (wave.planned_ideal_eight_worker_hours()),
                "maximum_raw_receipt_count": wave.MAXIMUM_RAW_RECEIPT_COUNT,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
