#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.generation1_successor_batch import (
    atomic_write_json,
    build_batch,
    build_readiness,
    canonical_bytes,
    validate_batch,
    validate_readiness,
)

DEFAULT_BATCH = REPO_ROOT / "configs/experiment/generation1_c3_successor_mechanisms_draft.json"
DEFAULT_READINESS = REPO_ROOT / "proof/GENERATION1_C3_SUCCESSOR_MECHANISMS_READINESS.json"


def _matches(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return canonical_bytes(actual) == canonical_bytes(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-out", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--readiness-out", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    batch = build_batch()
    validate_batch(batch)
    readiness = build_readiness(batch)
    validate_readiness(readiness, batch)
    if args.check:
        if not _matches(args.batch_out, batch) or not _matches(args.readiness_out, readiness):
            raise SystemExit("successor batch artifacts are missing or stale")
    else:
        atomic_write_json(args.batch_out, batch)
        atomic_write_json(args.readiness_out, readiness)
    print(
        json.dumps(
            {
                "batch": str(args.batch_out),
                "batch_sha256": batch["batch_sha256"],
                "readiness": str(args.readiness_out),
                "readiness_sha256": readiness["readiness_sha256"],
                "study_count": readiness["study_count"],
                "execution_ready": readiness["execution_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
