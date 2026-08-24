"""Focused tests for the write-once Odyssey clean-clone receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import odyssey_clean_clone as clean_clone


def test_clean_clone_scope_includes_all_execution_and_preflight_producers() -> None:
    assert "src/substrate/odyssey_detachment.py" in clean_clone.LINT_TARGETS
    assert "tests/substrate/test_odyssey_detachment.py" in clean_clone.TEST_TARGETS
    assert "src/substrate/odyssey_model_canary.py" in clean_clone.LINT_TARGETS
    assert "tests/substrate/test_odyssey_model_canary.py" in clean_clone.TEST_TARGETS
    assert "src/substrate/odyssey_arms.py" in clean_clone.LINT_TARGETS
    assert "tests/substrate/test_odyssey_arms.py" in clean_clone.TEST_TARGETS
    assert "src/substrate/odyssey_rehearsal.py" in clean_clone.LINT_TARGETS
    assert "tests/substrate/test_odyssey_rehearsal.py" in clean_clone.TEST_TARGETS
    assert "src/substrate/odyssey_machine_subjects.py" in clean_clone.LINT_TARGETS
    assert "tests/substrate/test_odyssey_machine_subjects.py" in clean_clone.TEST_TARGETS


def _write_frozen(root: Path, inputs: dict[str, Path], implementation: dict[str, Path]) -> dict:
    body = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "activation": False,
        "input_sha256": {name: clean_clone.file_digest(path) for name, path in inputs.items()},
        "implementation_sha256": {name: clean_clone.file_digest(path) for name, path in implementation.items()},
    }
    body["sha256"] = clean_clone.digest(body)
    path = root / clean_clone.PLAN / "ODYSSEY_FROZEN_BUILD.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return body


def test_frozen_build_requires_current_input_and_implementation_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "plan.json"
    implementation_path = tmp_path / "source.py"
    input_path.write_text("plan", encoding="utf-8")
    implementation_path.write_text("source", encoding="utf-8")
    inputs = {"input": input_path}
    implementation = {"implementation": implementation_path}
    frozen = _write_frozen(tmp_path, inputs, implementation)
    monkeypatch.setattr(clean_clone.odyssey_transition, "build_inputs", lambda _root: inputs)
    monkeypatch.setattr(clean_clone.odyssey_transition, "implementation_inputs", lambda _root: implementation)

    assert clean_clone._frozen_build(tmp_path)["sha256"] == frozen["sha256"]
    implementation_path.write_text("drift", encoding="utf-8")
    with pytest.raises(clean_clone.Refused, match="implementation drifts"):
        clean_clone._frozen_build(tmp_path)


def test_receipt_writer_is_self_digested_and_write_once(tmp_path: Path) -> None:
    receipt = {
        "schema": "SUBSTRATE_ODYSSEY_CLEAN_CLONE_CI/v1",
        "program": clean_clone.PROGRAM,
        "activation": False,
        "all_pass": True,
    }
    receipt["sha256"] = clean_clone.digest(receipt)
    path = tmp_path / "receipt.json"

    clean_clone._write_json(path, receipt)
    assert clean_clone._read_json(path)["sha256"] == receipt["sha256"]
    with pytest.raises(clean_clone.Refused, match="overwrite"):
        clean_clone._write_json(path, receipt)
