"""Post-run recovery aggregation for the source-bound Generation 1 cognitive census.

The 3,003 scientific attempt receipts bind the complete byte hash of
``generation1_cognitive_corpus.py``. Changing that producer after the run would correctly invalidate
every attempt. This module therefore leaves the original producer byte-identical, delegates the
scientific aggregation to it, and adds only an operational classification of invalid receipt
directories to the resulting sealed corpus.

An invalid attempt is superseded only when a strictly later numbered attempt for the same
``(outer seed, experiment)`` cell has a valid self-sealed receipt. Everything else is unresolved and
keeps the recovery gate closed. The independent verifier implements the same rule separately.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .generation1_cognitive_corpus import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    DEFAULT_RUN_ROOT,
    build_corpus,
    canonical_sha256,
)


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == canonical_sha256(core)


def _successful_receipt(payload: dict[str, Any]) -> bool:

    worker = payload.get("worker_report")
    return bool(
        _valid_seal(payload, "attempt_sha256")
        and payload.get("returncode") == 0
        and payload.get("timed_out") is False
        and isinstance(payload.get("manifest"), dict)
        and isinstance(worker, dict)
        and isinstance(worker.get("manifest"), dict)
    )


def classify_attempt_receipts(run_root: Path) -> dict[str, int]:

    attempts = sorted(run_root.glob("seed_*/classes/*/attempt_[0-9][0-9][0-9]"))
    invalid: list[tuple[tuple[str, str], int]] = []
    maximum_valid_number: dict[tuple[str, str], int] = {}
    valid_count = 0
    for attempt in attempts:
        cell = (attempt.parents[2].name, attempt.parent.name)
        number = int(attempt.name.rsplit("_", 1)[-1])
        receipt_path = attempt / "attempt_receipt.json"
        try:
            if receipt_path.is_symlink():
                raise OSError("attempt receipt is a symlink")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append((cell, number))
            continue
        if not isinstance(receipt, dict) or not _valid_seal(receipt, "attempt_sha256"):
            invalid.append((cell, number))
            continue
        valid_count += 1
        if _successful_receipt(receipt):
            maximum_valid_number[cell] = max(number, maximum_valid_number.get(cell, -1))

    superseded = sum(
        number < maximum_valid_number.get(cell, -1) for cell, number in invalid
    )
    unresolved = len(invalid) - superseded
    return {
        "attempt_directory_count": len(attempts),
        "valid_attempt_receipt_count": valid_count,
        "invalid_attempt_receipt_count": len(invalid),
        "superseded_invalid_attempt_count": superseded,
        "unresolved_invalid_attempt_count": unresolved,
    }


def build_recovered_corpus(config_path: Path, run_root: Path) -> dict[str, Any]:

    corpus = build_corpus(config_path, run_root)
    core = {key: value for key, value in corpus.items() if key != "corpus_sha256"}
    operational = dict(core.get("operational_summary") or {})
    classified = classify_attempt_receipts(run_root)
    for field in (
        "attempt_directory_count",
        "valid_attempt_receipt_count",
        "invalid_attempt_receipt_count",
    ):
        if operational.get(field) != classified[field]:
            raise ValueError(f"original and recovery operational counts disagree for {field}")
    operational.update(
        {
            "superseded_invalid_attempt_count": classified[
                "superseded_invalid_attempt_count"
            ],
            "unresolved_invalid_attempt_count": classified["unresolved_invalid_attempt_count"],
        }
    )
    core["operational_summary"] = operational
    return {**core, "corpus_sha256": canonical_sha256(core)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    corpus = build_recovered_corpus(arguments.config.resolve(), arguments.run_root.resolve())
    _atomic_json(arguments.out.resolve(), corpus)
    print(
        json.dumps(
            {
                "corpus_complete": corpus["corpus_complete"],
                "complete_experiment_count": corpus["complete_experiment_count"],
                "eligible_experiment_count": corpus["eligible_experiment_count"],
                "operational_summary": corpus["operational_summary"],
                "output": str(arguments.out),
            },
            indent=2,
        )
    )
    return 0 if corpus["corpus_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
