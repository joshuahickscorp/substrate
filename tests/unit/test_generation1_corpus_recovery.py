from __future__ import annotations

from pathlib import Path
from typing import Any

import mop.studies.generation1_corpus_recovery as recovery
from mop.studies.generation1_cognitive_corpus import ATTEMPT_SCHEMA, canonical_bytes


def _write_receipt(path: Path, *, successful: bool) -> None:
    core: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "returncode": 0 if successful else 1,
        "timed_out": False,
        "manifest": {"path": "manifest.json", "sha256": "a" * 64},
        "worker_report": {"manifest": {"path": "manifest.json", "sha256": "a" * 64}},
    }
    receipt = {**core, "attempt_sha256": recovery.canonical_sha256(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(receipt) + b"\n")


def test_classification_requires_a_later_successful_receipt(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    (root / "seed_1/classes/cell_a/attempt_001").mkdir(parents=True)
    _write_receipt(
        root / "seed_1/classes/cell_a/attempt_002/attempt_receipt.json",
        successful=True,
    )
    (root / "seed_1/classes/cell_b/attempt_001").mkdir(parents=True)
    _write_receipt(
        root / "seed_1/classes/cell_b/attempt_002/attempt_receipt.json",
        successful=False,
    )

    result = recovery.classify_attempt_receipts(root)

    assert result == {
        "attempt_directory_count": 4,
        "valid_attempt_receipt_count": 2,
        "invalid_attempt_receipt_count": 2,
        "superseded_invalid_attempt_count": 1,
        "unresolved_invalid_attempt_count": 1,
    }


def test_recovery_adds_operational_breakdown_and_reseals(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_root = tmp_path / "runs"
    (run_root / "seed_1/classes/cell_a/attempt_001").mkdir(parents=True)
    _write_receipt(
        run_root / "seed_1/classes/cell_a/attempt_002/attempt_receipt.json",
        successful=True,
    )
    original_core = {
        "schema": "mop-generation1-cognitive-corpus/v2",
        "corpus_complete": True,
        "operational_summary": {
            "attempt_directory_count": 2,
            "valid_attempt_receipt_count": 1,
            "invalid_attempt_receipt_count": 1,
        },
    }
    monkeypatch.setattr(
        recovery,
        "build_corpus",
        lambda _config, _root: {
            **original_core,
            "corpus_sha256": recovery.canonical_sha256(original_core),
        },
    )

    result = recovery.build_recovered_corpus(tmp_path / "config.json", run_root)

    assert result["operational_summary"]["superseded_invalid_attempt_count"] == 1
    assert result["operational_summary"]["unresolved_invalid_attempt_count"] == 0
    sealed_core = {key: value for key, value in result.items() if key != "corpus_sha256"}
    assert result["corpus_sha256"] == recovery.canonical_sha256(sealed_core)
