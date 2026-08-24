"""Focused provenance checks for a completed R2 continuity lane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import r2_provenance_verifier as verifier


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(schema: str, payload: dict, *, status: str) -> dict:
    body = {
        "schema": schema,
        "program": verifier.PROGRAM,
        "scientific_status": status,
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body["sha256"] = verifier.digest(body)
    return body


def _install(root: Path, *, source_digest: str = "a" * 64, head: str = "fixture-head") -> dict[str, Path]:
    evidence = root / verifier.EVIDENCE
    result_path = evidence / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json"
    result = _sealed(
        "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT",
        {"source_digest": source_digest},
        status="complete",
    )
    _write(result_path, result)
    clean_path = evidence / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json"
    clean = _sealed(
        "SUBSTRATE_SANDBOX_CLEAN_CLONE",
        {
            "source_digest": source_digest,
            "all_pass": True,
            "checkout": {"all_pass": True, "head": head},
        },
        status="pass",
    )
    _write(clean_path, clean)
    return {"result": result_path, "clean": clean_path, "output": evidence / "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION.json"}


def _matching_environment(monkeypatch: pytest.MonkeyPatch, *, source: str = "a" * 64, head: str = "fixture-head") -> None:
    monkeypatch.setattr(verifier, "_campaign_source_digest", lambda _root: source)
    monkeypatch.setattr(verifier, "_git_head", lambda _root: head)


def test_provenance_receipt_binds_result_clean_clone_and_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _install(tmp_path)
    _matching_environment(monkeypatch)

    receipt = verifier.verify(tmp_path, paths["output"])

    assert receipt["scientific_status"] == "pass"
    assert receipt["independently_verified"] is True
    assert receipt["git_head"] == "fixture-head"
    assert receipt["longitudinal_result"]["source_digest"] == "a" * 64
    assert receipt["clean_clone"]["source_digest"] == "a" * 64
    assert verifier._read_json(paths["output"], require_digest=True)["sha256"] == receipt["sha256"]


def test_provenance_refuses_current_source_or_clean_clone_head_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _install(tmp_path)
    _matching_environment(monkeypatch, source="b" * 64)
    with pytest.raises(verifier.Refused, match="source_digest"):
        verifier.verify(tmp_path, paths["output"])

    _matching_environment(monkeypatch)
    monkeypatch.setattr(verifier, "_git_head", lambda _root: "other-head")
    with pytest.raises(verifier.Refused, match="checkout.head"):
        verifier.verify(tmp_path, paths["output"])


def test_provenance_refuses_unsealed_or_source_mismatched_clean_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _install(tmp_path)
    _matching_environment(monkeypatch)
    clean = verifier._read_json(paths["clean"], require_digest=True)
    clean["source_digest"] = "b" * 64
    clean.pop("sha256")
    clean["sha256"] = verifier.digest(clean)
    _write(paths["clean"], clean)
    with pytest.raises(verifier.Refused, match="clean-clone source_digest"):
        verifier.verify(tmp_path, paths["output"])

    clean["sha256"] = "not-a-real-digest"
    paths["clean"].write_text(json.dumps(clean, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(verifier.Refused, match="self-digest mismatch"):
        verifier.verify(tmp_path, paths["output"])
