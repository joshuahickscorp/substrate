#!/usr/bin/env python3
"""Fail-closed sealed-boundary handoff to the adaptive parallel C2 verifier."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studio.generation1_supervisor import load_program, read_status

SOURCE_PROGRAM = REPO_ROOT / "configs/campaign/generation1_context_routing_adaptive25_labeled.json"
TARGET_PROGRAM = REPO_ROOT / "configs/campaign/generation1_context_routing_adaptive25_parallel_verify.json"
SHARD_IDS = tuple(f"g1_c2_context_routing_shard_{index:02d}" for index in range(4))


def _run(command: list[str]) -> None:
    print(json.dumps({"event": "command", "command": command}), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    arguments = parser.parse_args()
    if not 1.0 <= arguments.poll_seconds <= 60.0:
        raise SystemExit("poll seconds must be in [1, 60]")
    set_process_label("mop-c2-verify-handoff")
    source = load_program(SOURCE_PROGRAM)
    while True:
        status = read_status(source)
        state = status["state"]
        print(
            json.dumps(
                {
                    "event": "source-status",
                    "state": state,
                    "current_capsule": status.get("current_capsule"),
                    "problems": status.get("problems"),
                }
            ),
            flush=True,
        )
        if state == "drained":
            break
        if state in {"complete", "failure_hold", "integrity_hold"}:
            raise SystemExit(f"source campaign reached unexpected terminal state {state}")
        time.sleep(arguments.poll_seconds)

    if status.get("problems"):
        raise SystemExit("source campaign drained with problems")
    capsules = status.get("capsules") or {}
    incomplete = [task_id for task_id in SHARD_IDS if capsules.get(task_id, {}).get("status") != "complete"]
    if incomplete:
        raise SystemExit(f"source campaign drained before sealed shard completion: {incomplete}")

    python = str(REPO_ROOT / ".venv/bin/python")
    _run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/unit/test_generation1_context_routing_verify_parallel.py",
        ]
    )
    _run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/unit/test_local_execution_throttle.py",
            "-k",
            "opportunistic_profile_claims_sub_95_percent_gaps_and_bounds_v5_memory",
        ]
    )
    _run([python, "scripts/build_generation1_context_routing_parallel_verify_program.py", "--check"])
    _run(
        [
            python,
            "scripts/mop_generation1_campaign.py",
            "validate",
            "--program",
            str(TARGET_PROGRAM),
        ]
    )
    _run(
        [
            python,
            "scripts/mop_generation1_campaign.py",
            "start",
            "--program",
            str(TARGET_PROGRAM),
            "--execute",
        ]
    )
    target = load_program(TARGET_PROGRAM)
    target_status = read_status(target)
    if target_status["state"] not in {"running", "resource_wait"}:
        raise SystemExit(f"target campaign failed to enter a live state: {target_status['state']}")
    print(
        json.dumps(
            {
                "event": "handoff-complete",
                "target_state": target_status["state"],
                "current_capsule": target_status.get("current_capsule"),
                "problems": target_status.get("problems"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
