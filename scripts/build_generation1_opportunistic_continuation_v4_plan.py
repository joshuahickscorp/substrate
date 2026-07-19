#!/usr/bin/env python3
"""Build the sealed opportunistic recovery-to-synthesis continuation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studio.generation1_continuation import PLAN_SCHEMA, load_plan
from mop.studio.generation1_supervisor import (
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
)

RECOVERY_PROGRAM = (
    REPO_ROOT / "configs/campaign/generation1_empirical_recovery_v4_opportunistic.json"
)
SYNTHESIS_PROGRAM = (
    REPO_ROOT
    / "configs/campaign/generation1_evidence_synthesis_recovery_v4_opportunistic.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "configs/campaign/generation1_recovery_continuation_v4_opportunistic.json"
)
ROUTER_ID = "generation1-recovery-to-synthesis-v4-opportunistic"
OUT_DIR = f"runs/generation1/{ROUTER_ID}"


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _authority(path: str) -> dict[str, str]:
    source = (REPO_ROOT / path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"continuation authority must be a regular file: {source}")
    return {"path": path, "sha256": sha256_file(source)}


def _program_reference(path: Path) -> dict[str, str]:
    program = load_program(path)
    return {
        "path": _repo_path(path),
        "file_sha256": sha256_file(path),
        "program_sha256": program.program_sha256,
    }


def build_plan() -> dict[str, object]:
    core = {
        "schema": PLAN_SCHEMA,
        "router_id": ROUTER_ID,
        "out_dir": OUT_DIR,
        "prerequisite": _program_reference(RECOVERY_PROGRAM),
        "target": _program_reference(SYNTHESIS_PROGRAM),
        "authorities": [
            _authority("scripts/build_generation1_opportunistic_continuation_v4_plan.py"),
            _authority("scripts/mop_generation1_continuation.py"),
            _authority("src/mop/studio/generation1_continuation.py"),
        ],
        "control": {"poll_seconds": 30, "startup_ack_seconds": 120},
    }
    return {**core, "plan_sha256": canonical_sha256(core)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit("continuation plan must remain inside the repository")
    plan = build_plan()
    if arguments.check:
        if json.loads(output.read_text(encoding="utf-8")) != plan:
            raise SystemExit("Generation 1 continuation plan is stale")
    else:
        atomic_write_json(output, plan)
    loaded = load_plan(output)
    if loaded.plan_sha256 != plan["plan_sha256"]:
        raise ValueError("loaded continuation digest differs from generated digest")
    print(
        json.dumps(
            {
                "path": str(output),
                "router_id": ROUTER_ID,
                "plan_sha256": plan["plan_sha256"],
                "prerequisite_program_sha256": plan["prerequisite"]["program_sha256"],
                "target_program_sha256": plan["target"]["program_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
