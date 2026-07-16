#!/usr/bin/env python3
"""Build a fresh three-capsule recovery epoch for the completed Generation 1 census data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_supervisor import (
    PROGRAM_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
)
from mop.studio.local_throttle import load_policy

BASE_PROGRAM = REPO_ROOT / "configs/campaign/generation1_empirical_program_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "configs/campaign/generation1_empirical_recovery_v1.json"
PROGRAM_ID = "generation1-empirical-cognitive-corpus-v2-recovery1"
PROGRAM_ROOT = f"runs/generation1/{PROGRAM_ID}"
CAPSULE_IDS = (
    "g1_cognitive_corpus_aggregate",
    "g1_cognitive_corpus_verify",
    "g1_empirical_report",
)


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"recovery authority must be a regular file: {source}")
    return {"path": path, "sha256": sha256_file(source)}


def _reseal_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    core = {key: value for key, value in capsule.items() if key != "capsule_sha256"}
    return {**core, "capsule_sha256": canonical_sha256(core)}


def _refresh_authorities(capsule: dict[str, Any], paths: list[str] | None = None) -> None:
    selected = paths or [str(row["path"]) for row in capsule["authorities"]]
    capsule["authorities"] = [_authority(path) for path in selected]


def build_program() -> dict[str, Any]:
    base = json.loads(BASE_PROGRAM.read_text(encoding="utf-8"))
    by_id = {str(row["id"]): row for row in base["capsules"]}
    aggregate = copy.deepcopy(by_id[CAPSULE_IDS[0]])
    verify = copy.deepcopy(by_id[CAPSULE_IDS[1]])
    report = copy.deepcopy(by_id[CAPSULE_IDS[2]])

    aggregate["depends_on"] = []
    aggregate["command"] = [
        ".venv/bin/python",
        "scripts/generation1_recovery/generation1_cognitive_corpus.py",
        "--config",
        "configs/experiment/generation1_cognitive_corpus.json",
        "--run-root",
        "runs/generation1/cognitive_corpus",
        "--out",
        "proof/GENERATION1_COGNITIVE_CORPUS.json",
    ]
    fields = aggregate["artifacts"][0]["fields"]
    fields.pop("operational_summary.invalid_attempt_receipt_count", None)
    fields["operational_summary.unresolved_invalid_attempt_count"] = 0
    _refresh_authorities(
        aggregate,
        [
            "scripts/generation1_recovery/generation1_cognitive_corpus.py",
            "src/mop/studies/generation1_corpus_recovery.py",
            "src/mop/studies/generation1_cognitive_corpus.py",
            "configs/experiment/generation1_cognitive_corpus.json",
        ],
    )

    verify["depends_on"] = [aggregate["id"]]
    verify_fields = verify["artifacts"][0]["fields"]
    verify_fields["attempt_audit.superseded_invalid_count"] = 8
    verify_fields["attempt_audit.unresolved_invalid_count"] = 0
    verify_fields["authority_audit.scientific_fingerprint_recompute"] = {
        "selected_cell_count": 3003,
        "mode_counts": {
            "canonical_json": 2763,
            "legacy_pre_json_integer_keys": 240,
        },
    }
    _refresh_authorities(verify)

    report["depends_on"] = [verify["id"]]
    _refresh_authorities(report)

    capsules = [_reseal_capsule(row) for row in (aggregate, verify, report)]
    authority_paths = sorted(
        {
            "scripts/build_generation1_recovery_program.py",
            *(str(item["path"]) for row in capsules for item in row["authorities"]),
        }
    )
    core = {
        "schema": PROGRAM_SCHEMA,
        "program_id": PROGRAM_ID,
        "program_root": PROGRAM_ROOT,
        "policy": _authority("configs/local_execution_throttle.yaml"),
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
        raise ValueError("loaded recovery program digest differs from the generated digest")
    policy = load_policy(REPO_ROOT / program.policy.path)
    hard_wall_minutes = int(policy.limits["hard_wall_minutes"])
    known_markers = {str(value) for value in policy.monitor["known_heavy_markers"]}
    problems: list[str] = []
    for capsule in program.capsules:
        if capsule.resources.process_marker not in known_markers:
            problems.append(
                f"{capsule.capsule_id}: unknown marker {capsule.resources.process_marker}"
            )
        problems.extend(
            f"{capsule.capsule_id}: {problem}"
            for problem in capsule.task_declaration().validate(hard_wall_minutes)
        )
    if problems:
        raise ValueError("recovery program is not runtime-admissible:\n" + "\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("recovery program manifest must remain inside the repository")
    program = build_program()
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != program:
            raise SystemExit("Generation 1 recovery program manifest is stale")
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
