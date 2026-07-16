#!/usr/bin/env python3
"""Run or check the sealed G1-C3 V1/M1 communication mechanics pilot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.generation1_c3_communication import (
    build_config,
    run_pilot,
    validate_result,
)

DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_C3_COMMUNICATION_PILOT.json"


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--v1-seed-start", type=int, default=20_278_001)
    parser.add_argument("--m1-seed-start", type=int, default=20_279_001)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        try:
            result = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"communication pilot artifact is missing or malformed: {exc}") from exc
        validate_result(result)
    else:
        config = build_config(
            seed_count=args.seed_count,
            v1_seed_start=args.v1_seed_start,
            m1_seed_start=args.m1_seed_start,
        )
        result = run_pilot(config)
        validate_result(result)
        _atomic_write_json(args.output, result)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_sha256": result["result_sha256"],
                "seed_count_per_lane": result["lanes"]["G1-V1"]["null"]["seed_count"],
                "lane_pilot_discrimination": result["decision"]["lane_pilot_discrimination"],
                "scientific_confirmation": result["decision"]["scientific_confirmation"],
                "activation_allowed": result["activation_allowed"],
                "scientific_promotion": result["scientific_promotion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
