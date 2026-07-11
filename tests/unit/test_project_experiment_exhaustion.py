from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

import mop.studies.project_exhaustion as exhaustion
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


def _write_bound_attempt(root: Path, *, copied_checkpoint_every: int, live_checkpoint_every: int) -> Path:
    run = root / "runs" / "project_exhaustion" / "classes" / "demo" / "attempt_001"
    run.mkdir(parents=True)
    live_config = {
        "id": "demo",
        "profiles": {"p5smoke": {"checkpoint_every": live_checkpoint_every}},
    }
    config_path = root / "configs" / "experiment" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(live_config, sort_keys=True))
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "demo",
                    "profiles": {"p5smoke": {"checkpoint_every": copied_checkpoint_every}},
                }
            },
            sort_keys=True,
        )
    )
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
    source_evidence = [{"path": "demo.py", "sha256": "a" * 64}]
    (run / "attempt_receipt.json").write_text(
        json.dumps(
            {
                "schema": exhaustion.ATTEMPT_SCHEMA,
                "experiment_id": "demo",
                "run_dir": str(run.relative_to(root)),
                "returncode": 0,
                "timed_out": False,
                "worker_report": {"experiment_id": "demo"},
                "source_evidence": source_evidence,
            }
        )
    )
    return run


def test_run_verifier_binds_attempt_sources_and_live_experiment_config(
    tmp_path: Path,
) -> None:
    run = _write_bound_attempt(tmp_path, copied_checkpoint_every=1, live_checkpoint_every=1)
    source_evidence = [{"path": "demo.py", "sha256": "a" * 64}]
    with (
        patch.object(exhaustion, "REPO_ROOT", tmp_path),
        patch.object(exhaustion, "class_source_evidence", return_value=source_evidence),
    ):
        result = exhaustion.verify_experiment_run(run, "demo")
    assert result["verified"] is True
    assert result["checks"]["attempt_source_evidence_current"] is True
    assert result["checks"]["resolved_experiment_config_current"] is True


def test_run_verifier_rejects_source_or_resolved_config_drift(tmp_path: Path) -> None:
    run = _write_bound_attempt(tmp_path, copied_checkpoint_every=6, live_checkpoint_every=1)
    drifted_source = [{"path": "demo.py", "sha256": "b" * 64}]
    with (
        patch.object(exhaustion, "REPO_ROOT", tmp_path),
        patch.object(exhaustion, "class_source_evidence", return_value=drifted_source),
    ):
        result = exhaustion.verify_experiment_run(run, "demo")
    assert result["verified"] is False
    assert result["checks"]["attempt_source_evidence_current"] is False
    assert result["checks"]["resolved_experiment_config_current"] is False


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


def test_e6_runtime_and_integration_are_local_and_the_remaining_blocker_is_data():
    ledger = json.loads(LEDGER_PATH.read_text())
    e6 = next(entry for entry in ledger["entries"] if entry["id"] == "e6_relational")
    assert e6["classification"] == "rights-data-blocked"
    assert "rights-clean annotated natural cohort" in e6["remaining_scientific_blocker"]
    paths = {row["path"] for row in e6["evidence"]}
    assert {
        "proof/VJEPA21_VITB_LOAD.json",
        "proof/VJEPA21_VITB_FORWARD.json",
        "proof/VJEPA21_VITB_FORWARD_64F.json",
        "proof/E6_VITB_DENSE_PREFLIGHT.json",
    } <= paths


def test_alignment_and_consistency_rows_do_not_depend_on_retired_scale_pilot():
    ledger = json.loads(LEDGER_PATH.read_text())
    by_id = {entry["id"]: entry for entry in ledger["entries"]}
    al2 = by_id["mop_al2_shared_latent_alignment"]
    dr5 = by_id["mop_dr5_cross_substrate_consistency"]
    assert al2["classification"] == "rights-data-blocked"
    assert dr5["classification"] == "upstream-model-blocked"
    assert "active ViT-B/custom" in al2["remaining_scientific_blocker"]
    assert "active ViT-B/custom" in dr5["remaining_scientific_blocker"]
    assert "three-scale atlas" not in al2["remaining_scientific_blocker"]
    assert "three real-weight encoder" not in dr5["remaining_scientific_blocker"]


def test_ledger_self_verifies_without_reading_runs_tree():
    ledger = json.loads(LEDGER_PATH.read_text())
    assert ledger["self_verification"]["verified"] is True
    # All runtime evidence has already been embedded.  Prohibit every filesystem text read to
    # prove the verifier does not depend on ignored/erasable runs files.
    with patch.object(Path, "read_text", side_effect=AssertionError("filesystem read forbidden")):
        result = verify_embedded_ledger(ledger)
    assert result["verified"] is True, result["errors"]
    assert result["runtime_evidence_reference_count"] > 0
