"""Focused isolation and source-binding tests for the Odyssey G12 runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from substrate import odyssey_mutations as mutations


def _run(root: Path, *arguments: str) -> None:
    completed = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _fixture_transition() -> str:
    return """from pathlib import Path

PROGRAM = "substrate-odyssey-r2-handoff-v1"
PLAN = Path("plans/substrate/tangible_next_launch")


def build_inputs(root: Path) -> dict[str, Path]:
    return {
        "hardened_design": root / PLAN / "ODYSSEY_7D.hardened.draft.json",
        "autopivot_policy": root / PLAN / "R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "task_bank": root / PLAN / "ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        "resource_calibration": root / PLAN / "RESOURCE_CALIBRATION_SPEC.draft.json",
        "shared_storage": root / PLAN / "ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "frontier_contract": root / PLAN / "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "rendered_build_index": root / PLAN / "frontiers/FRONTIER_BUILD_INDEX.json",
        "source_selection_template": root / PLAN / "ODYSSEY_SOURCE_SELECTION.template.json",
        "public_model_canary_template": root / PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        "human_evidence_pack_template": root / PLAN / "ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
        "operator_decision": root / "operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
    }


def implementation_inputs(root: Path) -> dict[str, Path]:
    return {
        "transition_controller": root / "src/substrate/odyssey_transition.py",
        "frontier_renderer": root / "src/substrate/odyssey7d.py",
        "task_bank_generator": root / "src/substrate/odyssey_task_bank.py",
        "manifest_materializer": root / "src/substrate/odyssey_manifest_materializer.py",
        "odyssey_arms": root / "src/substrate/odyssey_arms.py",
        "odyssey_worker": root / "src/substrate/odyssey_worker.py",
        "odyssey_authority": root / "src/substrate/odyssey_authority.py",
        "public_model_canary": root / "src/substrate/odyssey_model_canary.py",
        "odyssey_clean_clone": root / "src/substrate/odyssey_clean_clone.py",
        "odyssey_detachment": root / "src/substrate/odyssey_detachment.py",
        "telegram_probe": root / "src/substrate/odyssey_telegram_probe.py",
        "telegram_notifier": root / "tools/odyssey7d_telegram_notifier.py",
        "r2_continuity_verifier": root / "src/substrate/r2_continuity_verifier.py",
        "r2_provenance_verifier": root / "src/substrate/r2_provenance_verifier.py",
        "odyssey_mutations": root / "src/substrate/odyssey_mutations.py",
    }
"""


def _fixture_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a small committed repository that still runs the real guards."""
    repository = Path(__file__).parents[2]
    root = tmp_path / "odyssey-mutation-fixture"
    _copy(repository / "src/substrate/__init__.py", root / "src/substrate/__init__.py")
    for filename in (
        "odyssey_mutations.py",
        "odyssey_authority.py",
        "odyssey_model_canary.py",
        "odyssey_density.py",
        "odyssey7d.py",
        "odyssey_arms.py",
        "odyssey_worker.py",
        "odyssey_task_bank.py",
        "odyssey_manifest_materializer.py",
        "odyssey_clean_clone.py",
        "odyssey_detachment.py",
        "odyssey_telegram_probe.py",
        "r2_continuity_verifier.py",
        "r2_provenance_verifier.py",
    ):
        _copy(repository / "src/substrate" / filename, root / "src/substrate" / filename)
    transition = root / "src/substrate/odyssey_transition.py"
    transition.parent.mkdir(parents=True, exist_ok=True)
    transition.write_text(_fixture_transition(), encoding="utf-8")
    _copy(repository / "tools/odyssey7d_telegram_notifier.py", root / "tools/odyssey7d_telegram_notifier.py")
    plan = repository / "plans/substrate/tangible_next_launch"
    for filename in (
        "ODYSSEY_7D.hardened.draft.json",
        "R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        "RESOURCE_CALIBRATION_SPEC.draft.json",
        "ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "ODYSSEY_SOURCE_SELECTION.template.json",
        "ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        "ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
        "frontiers/FRONTIER_BUILD_INDEX.json",
        # Frontier-F candidate generation needs the committed clip index; the
        # 90 GB audio corpus is intentionally absent from mutation fixtures.
        "LIBRISPEECH_CLIP_INDEX.json",
    ):
        _copy(plan / filename, root / "plans/substrate/tangible_next_launch" / filename)
    _copy(
        repository / "operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
        root / "operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
    )

    fixture_paths = {
        "transition_controller": root / "src/substrate/odyssey_transition.py",
        "frontier_renderer": root / "src/substrate/odyssey7d.py",
        "task_bank_generator": root / "src/substrate/odyssey_task_bank.py",
        "manifest_materializer": root / "src/substrate/odyssey_manifest_materializer.py",
        "odyssey_arms": root / "src/substrate/odyssey_arms.py",
        "odyssey_worker": root / "src/substrate/odyssey_worker.py",
        "odyssey_authority": root / "src/substrate/odyssey_authority.py",
        "public_model_canary": root / "src/substrate/odyssey_model_canary.py",
        "odyssey_clean_clone": root / "src/substrate/odyssey_clean_clone.py",
        "odyssey_detachment": root / "src/substrate/odyssey_detachment.py",
        "telegram_probe": root / "src/substrate/odyssey_telegram_probe.py",
        "telegram_notifier": root / "tools/odyssey7d_telegram_notifier.py",
        "r2_continuity_verifier": root / "src/substrate/r2_continuity_verifier.py",
        "r2_provenance_verifier": root / "src/substrate/r2_provenance_verifier.py",
        "odyssey_mutations": root / "src/substrate/odyssey_mutations.py",
    }
    fixture_inputs = {
        "hardened_design": root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.hardened.draft.json",
        "autopivot_policy": root / "plans/substrate/tangible_next_launch/R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "task_bank": root / "plans/substrate/tangible_next_launch/ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        "resource_calibration": root / "plans/substrate/tangible_next_launch/RESOURCE_CALIBRATION_SPEC.draft.json",
        "shared_storage": root / "plans/substrate/tangible_next_launch/ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "frontier_contract": root / "plans/substrate/tangible_next_launch/ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "rendered_build_index": root / "plans/substrate/tangible_next_launch/frontiers/FRONTIER_BUILD_INDEX.json",
        "source_selection_template": root / "plans/substrate/tangible_next_launch/ODYSSEY_SOURCE_SELECTION.template.json",
        "public_model_canary_template": root / "plans/substrate/tangible_next_launch/ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        "human_evidence_pack_template": root / "plans/substrate/tangible_next_launch/ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
        "operator_decision": root / "operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
    }
    frozen = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "program": mutations.odyssey_transition.PROGRAM,
        "activation": False,
        "scientific_status": "frozen_waiting_for_verified_r2",
        "input_sha256": {name: mutations.file_digest(path) for name, path in fixture_inputs.items()},
        "implementation_sha256": {name: mutations.file_digest(path) for name, path in fixture_paths.items()},
    }
    frozen["sha256"] = mutations.digest(frozen)
    frozen_path = root / mutations.PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, sort_keys=True), encoding="utf-8")

    _run(root, "init", "--quiet")
    _run(root, "config", "user.email", "fixture@example.invalid")
    _run(root, "config", "user.name", "Odyssey Fixture")
    _run(root, "add", ".")
    _run(root, "commit", "--quiet", "-m", "freeze mutation fixture")
    monkeypatch.setattr(
        mutations.odyssey_transition,
        "implementation_inputs",
        lambda source_root: {name: source_root / path.relative_to(root) for name, path in fixture_paths.items()},
    )
    monkeypatch.setattr(
        mutations.odyssey_transition, "build_inputs", lambda source_root: {name: source_root / path.relative_to(root) for name, path in fixture_inputs.items()}
    )
    return root


def test_g12_runner_uses_exact_clone_and_rejects_all_control_plane_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from substrate import odyssey_authority as authority

    root = _fixture_root(tmp_path, monkeypatch)
    report_path = root / "evidence/substrate/odyssey/mutations/G12.fixture.json"

    report = mutations.run(root, report_path, python_executable=sys.executable)

    assert report["schema"] == mutations.SCHEMA
    assert report["all_pass"] is True
    assert report["scientific_evidence"] is False
    assert report["declared_mutation_count"] == len(mutations.MUTATIONS)
    assert report["injected_count"] == len(mutations.MUTATIONS)
    assert report["survivors"] == []
    assert report["clean_clone"]["exact_commit_checkout"] is True
    assert all(row["injected"] is True and row["detected"] is True and row["clean_case_passed"] is True for row in report["mutations"])
    assert all((root / row["clean_receipt"]["path"]).is_file() for row in report["mutations"])
    assert all((root / row["mutant_receipt"]["path"]).is_file() for row in report["mutations"])
    assert mutations._read_json(report_path, require_digest=True)["sha256"] == report["sha256"]
    frozen = mutations._read_json(root / mutations.PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    authority._gate_specific_checks(root, "G12", report, frozen)
    # Source mutations occurred only below temp fixtures, never in the sealed
    # source tree that produced the report.
    assert "isolated-odyssey-mutation" not in (root / "src/substrate/odyssey_worker.py").read_text(encoding="utf-8")


def test_g12_runner_refuses_when_its_own_source_is_not_in_frozen_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture_root(tmp_path, monkeypatch)
    frozen_path = root / mutations.PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["implementation_sha256"].pop(mutations.RUNNER_SOURCE_KEY)
    frozen.pop("sha256")
    frozen["sha256"] = mutations.digest(frozen)
    frozen_path.write_text(json.dumps(frozen, sort_keys=True), encoding="utf-8")

    with pytest.raises(mutations.Refused, match="bind odyssey_mutations"):
        mutations.run(root, root / "evidence/G12.json", python_executable=sys.executable)


def test_mutation_report_is_write_once(tmp_path: Path) -> None:
    report = {"schema": mutations.SCHEMA, "activation": False}
    report["sha256"] = mutations.digest(report)
    path = tmp_path / "G12.json"

    mutations._write_json(path, report)

    with pytest.raises(mutations.Refused, match="overwrite"):
        mutations._write_json(path, report)


def test_probe_refuses_to_write_synthetic_fixtures_in_an_unmarked_directory(tmp_path: Path) -> None:
    with pytest.raises(mutations.Refused, match="isolated fixture"):
        mutations.probe(tmp_path, mutations.MUTATIONS[0].identifier, "clean")
