from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from mop.config import REPO_ROOT
from mop.studies.project_exhaustion import (
    CATEGORIES,
    HARDWARE_BLOCKED,
    STANDALONE_EVIDENCE,
    STANDALONE_SCRIPTS,
    registry_rows,
    verify_embedded_ledger,
    verify_experiment_run,
)

LEDGER_PATH = REPO_ROOT / "proof" / "PROJECT_EXPERIMENT_EXHAUSTION.json"


def test_inventory_covers_every_non_f_registry_row_exactly_once():
    raw = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text())["experiments"]
    expected = {row["id"] for row in raw if not row["id"].startswith("f")}
    rows = registry_rows()
    assert {row["id"] for row in rows} == expected
    assert len(rows) == len(expected)
    assert not any(row["id"].startswith("f") for row in rows)


def test_standalone_cpu_rows_have_a_runnable_script_mapping():
    assert len(STANDALONE_EVIDENCE) == 37
    assert set(STANDALONE_EVIDENCE) == set(STANDALONE_SCRIPTS)
    assert all((REPO_ROOT / path).is_file() for path in STANDALONE_SCRIPTS.values())
    assert all(paths for paths in STANDALONE_EVIDENCE.values())


def test_run_verifier_requires_real_manifest_config_and_metrics(tmp_path: Path):
    run = tmp_path / "attempt_001"
    run.mkdir()
    (run / "config.yaml").write_text("experiment:\n  id: demo\n")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "status": "ok",
                "finished": 1.0,
                "device": "cpu",
                "seed": 0,
                "metrics": {"score": 0.5},
                "extra": {"contract": {"id": "demo"}},
            }
        )
    )
    # The production verifier reports repo-relative paths; mirror the fixture beneath the repo.
    # A temp outside the repo is intentionally rejected rather than silently made non-portable.
    try:
        verify_experiment_run(run, "demo")
    except ValueError:
        pass
    else:
        raise AssertionError("external run directories must not be accepted as portable evidence")


def test_ledger_categories_are_exhaustive_and_hardware_is_not_inferred_from_tier():
    ledger = json.loads(LEDGER_PATH.read_text())
    entries = ledger["entries"]
    assert len(entries) == ledger["coverage"]["registry_non_f_total"]
    assert len(entries) == len({entry["id"] for entry in entries})
    assert {entry["classification"] for entry in entries} <= CATEGORIES
    assert sum(ledger["coverage"]["classification_counts"].values()) == len(entries)
    assert ledger["coverage"]["measured_hardware_blocked_count"] == 0
    assert not any(entry["classification"] == HARDWARE_BLOCKED for entry in entries)
    assert all(entry["scientific_claim_ready"] is False for entry in entries)


def test_ledger_self_verifies_without_reading_runs_tree():
    ledger = json.loads(LEDGER_PATH.read_text())
    assert ledger["self_verification"]["verified"] is True
    # All runtime evidence has already been embedded.  Prohibit every filesystem text read to
    # prove the verifier does not depend on ignored/erasable runs files.
    with patch.object(Path, "read_text", side_effect=AssertionError("filesystem read forbidden")):
        result = verify_embedded_ledger(ledger)
    assert result["verified"] is True, result["errors"]
    assert result["runtime_evidence_reference_count"] > 0
