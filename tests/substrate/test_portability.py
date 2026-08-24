"""Tests for Substrate/Odyssey portability manifest, verify, and restore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import portability as P


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_root(tmp_path: Path) -> Path:
    """Build a tiny portable tree with one tool artifact and one corpus dataset."""
    root = tmp_path / "tree"
    (root / "plans" / "substrate" / "tangible_next_launch").mkdir(parents=True)
    (root / "src" / "substrate").mkdir(parents=True)
    (root / "data" / "substrate" / "tangible_sandbox" / "prefetch" / "odyssey-public-v1" / "demo").mkdir(parents=True)

    tool = root / "bin" / "demo-tool"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b"demo-tool-bytes-v1")
    tool.chmod(0o755)

    corpus_file = root / "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/demo/hello.txt"
    corpus_file.write_text("hello corpus\n", encoding="utf-8")
    digest = P.file_sha256(corpus_file)
    manifest = corpus_file.parent / "MANIFEST.sha256"
    manifest.write_text(f"{digest}  hello.txt\n", encoding="utf-8")

    frozen = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "sha256": "a" * 64,
        "activation": False,
    }
    sealed = {
        "schema": "SUBSTRATE_ODYSSEY_SOURCE_SELECTION/v1",
        "sha256": "b" * 64,
        "status": "sealed",
        "activation": False,
    }
    _write_json(root / P.FROZEN_BUILD_REL, frozen)
    _write_json(root / P.SOURCE_SELECTION_REL, sealed)
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    document = {
        "schema": P.SCHEMA,
        "program": P.PROGRAM,
        "generated_at": "2026-08-04T00:00:00Z",
        "activation": False,
        "repo": {
            "git_head": "deadbeef",
            "frozen_build_digest": "a" * 64,
            "source_selection_digest": "b" * 64,
        },
        "python": {
            "interpreter_version": "",
            "preferred_python": "3.12",
            "lockfile": "uv.lock",
            "lockfile_sha256": P.file_sha256(root / "uv.lock"),
            "resolved_dependencies": [],
            "install_extras": [],
        },
        "tools": [
            {
                "id": "demo_tool",
                "absolute_path": str(tool.resolve()),
                "repo_relative_path": "",
                "version": "1",
                "artifact_sha256": P.file_sha256(tool),
                "install_method": "test_fixture",
                "reinstall_command": "echo reinstall-demo-tool",
                "needs_human": True,
                "human_reason": "fixture",
                "in_repo": False,
                "present_at_generation": True,
            }
        ],
        "ollama_models": [],
        "corpus_roots": [
            {
                "dataset": "demo",
                "root_relative": "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/demo",
                "byte_size": corpus_file.stat().st_size,
                "manifest_sha256_path": "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/demo/MANIFEST.sha256",
                "has_integrity_manifest": True,
                "manifest_file_sha256": P.file_sha256(manifest),
            }
        ],
    }
    _write_json(root / P.MANIFEST_REL, document)
    return root


def test_file_sha256_changes_when_bytes_change(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"alpha")
    first = P.file_sha256(path)
    path.write_bytes(b"beta")
    second = P.file_sha256(path)
    assert first != second
    assert len(first) == 64
    assert len(second) == 64


def test_verify_reports_missing_when_tool_path_removed_from_view(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path)
    manifest = P.load_manifest(root=root)
    # Simulate: point the manifest at a path that does not exist without deleting the real tool.
    manifest["tools"][0]["absolute_path"] = str(tmp_path / "no-such-tool-binary")
    # And clear resolution fallbacks by using an unknown tool id with only that path.
    report = P.verify(root, manifest, quick_corpus=True)
    tool_items = [item for item in report["items"] if item.get("kind") == "tool" and item.get("id") == "demo_tool"]
    assert tool_items
    assert tool_items[0]["status"] == P.STATUS_MISSING
    assert "reinstall-demo-tool" in tool_items[0]["remediation"]
    assert report["ok"] is False


def test_verify_reports_present_but_drifted_when_digest_mismatches(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path)
    manifest = P.load_manifest(root=root)
    manifest["tools"][0]["artifact_sha256"] = "0" * 64
    report = P.verify(root, manifest, quick_corpus=True)
    tool_items = [item for item in report["items"] if item.get("kind") == "tool" and item.get("id") == "demo_tool"]
    assert tool_items
    assert tool_items[0]["status"] == P.STATUS_DRIFTED
    assert tool_items[0]["observed_sha256"]
    assert tool_items[0]["observed_sha256"] != "0" * 64
    assert report["ok"] is False


def test_verify_corpus_full_check_matches_manifest(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path)
    manifest = P.load_manifest(root=root)
    report = P.verify(root, manifest, quick_corpus=False)
    corpus = [item for item in report["items"] if item.get("kind") == "corpus"]
    assert corpus
    assert corpus[0]["status"] == P.STATUS_MATCHING


def test_restore_is_idempotent_for_print_only_and_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _minimal_root(tmp_path)
    manifest = P.load_manifest(root=root)

    # Avoid real uv/ollama work: pretend venv is already healthy and no models are required.
    monkeypatch.setattr(P, "_recreate_venv", lambda _root, _info: {
        "ok": True,
        "action": "recreate_venv",
        "executed": False,
        "skipped": True,
        "detail": "fixture",
        "print_commands": [],
    })
    monkeypatch.setattr(P, "_pull_missing_models", lambda _models: {
        "ok": True,
        "action": "ollama_pull",
        "executed": False,
        "pulled": [],
        "already_present": [],
        "failed": [],
        "print_commands": [],
    })

    first = P.restore(root, manifest, quick_corpus=False)
    second = P.restore(root, manifest, quick_corpus=False)
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["actions"][0]["skipped"] is True
    assert second["actions"][0]["skipped"] is True
    # Corpus verification results stable
    first_corpus = first["actions"][2]["results"]
    second_corpus = second["actions"][2]["results"]
    assert first_corpus[0]["status"] == P.STATUS_MATCHING
    assert second_corpus[0]["status"] == P.STATUS_MATCHING


def test_generate_manifest_records_real_tool_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _minimal_root(tmp_path)
    # Point a TOOL_SPECS entry at the fixture binary via path candidates.
    demo_spec = {
        "id": "demo_measured_tool",
        "install_method": "test_fixture",
        "path_candidates": [str((root / "bin" / "demo-tool").resolve())],
        "which_names": [],
        "version_argv": None,
        "reinstall_command": "echo reinstall",
        "needs_human": True,
        "human_reason": "fixture",
        "in_repo": False,
    }
    monkeypatch.setattr(P, "TOOL_SPECS", [demo_spec])
    monkeypatch.setattr(P, "PINNED_OLLAMA_MODELS", ())
    monkeypatch.setattr(P, "_git_head", lambda _root: "deadbeef")
    monkeypatch.setattr(P, "_measure_python", lambda _root: {
        "interpreter_version": "3.12.0",
        "lockfile_sha256": P.file_sha256(root / "uv.lock"),
        "resolved_dependencies": [],
        "install_extras": [],
        "preferred_python": "3.12",
    })

    document = P.generate_manifest(root)
    tools = {t["id"]: t for t in document["tools"]}
    assert "demo_measured_tool" in tools
    measured = tools["demo_measured_tool"]
    assert measured["present_at_generation"] is True
    assert measured["artifact_sha256"] == P.file_sha256(root / "bin" / "demo-tool")

    # Alter bytes; regenerating must yield a different digest.
    (root / "bin" / "demo-tool").write_bytes(b"demo-tool-bytes-v2")
    document2 = P.generate_manifest(root)
    measured2 = {t["id"]: t for t in document2["tools"]}["demo_measured_tool"]
    assert measured2["artifact_sha256"] != measured["artifact_sha256"]


def test_verify_passes_on_host_against_generated_manifest() -> None:
    """Integration: the checked-in/generated manifest must verify on this host.

    Uses quick_corpus for speed in CI-like runs; full corpus is exercised by the operator command.
    """
    root = P.repo_root()
    path = root / P.MANIFEST_REL
    if not path.is_file():
        pytest.skip("portability manifest not generated yet")
    # Host integration requires data/ and tools; skip cleanly only if generate never ran.
    report = P.verify(root, quick_corpus=True)
    # Allow reporting for debugging on failure
    if not report["ok"]:
        blockers = report.get("blockers") or []
        # If only repo git_head drifted (uncommitted worktree) treat repo identity as non-blocking for this test?
        # Contract: verify passes on this host as-is. Repo digests should match generation.
        pytest.fail(json.dumps(blockers, indent=2, default=str)[:4000])
    assert report["ok"] is True
    assert report["counts"][P.STATUS_MISSING] == 0
    assert report["counts"][P.STATUS_DRIFTED] == 0
