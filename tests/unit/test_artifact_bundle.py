import json

from mop.studio.artifact_bundle import build_artifact_index, preset_paths, write_artifact_index


def test_artifact_index_copies_untracked_small_receipts(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = root / "runs" / "wave" / "receipt.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"ok": True}) + "\n")
    index = build_artifact_index(
        ["runs/wave/receipt.json"],
        repo_root=root,
        copy_dir=root / "proof" / "bundle",
        require_durable=True,
    )
    artifact = index["artifacts"][0]
    assert index["all_ok"] is True
    assert artifact["copied"] is True
    assert artifact["durable"] is True
    assert (root / "proof" / "bundle" / "runs" / "wave" / "receipt.json").exists()


def test_artifact_index_fails_missing_artifacts_by_default(tmp_path):
    index = build_artifact_index(["missing.json"], repo_root=tmp_path)
    assert index["all_ok"] is False
    assert "missing artifact" in index["problems"][0]
    allowed = build_artifact_index(["missing.json"], repo_root=tmp_path, allow_missing=True)
    assert allowed["all_ok"] is True


def test_artifact_index_flags_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json")
    index = build_artifact_index([bad], repo_root=tmp_path)
    assert index["all_ok"] is False
    assert index["artifacts"][0]["json_ok"] is False
    assert any("invalid JSON" in p for p in index["problems"])


def test_artifact_index_refuses_oversized_copy(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "receipt.md"
    path.write_text("x" * 32)
    index = build_artifact_index(
        ["receipt.md"],
        repo_root=root,
        copy_dir=root / "proof" / "bundle",
        max_copy_bytes=8,
        require_durable=True,
    )
    assert index["all_ok"] is False
    assert index["artifacts"][0]["copied"] is False
    assert any("exceeds max_copy_bytes" in p for p in index["problems"])


def test_write_artifact_index_round_trips(tmp_path):
    receipt = tmp_path / "receipt.json"
    out = tmp_path / "index.json"
    receipt.write_text(json.dumps({"ok": True}))
    index = build_artifact_index([receipt], repo_root=tmp_path)
    write_artifact_index(index, out)
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == "mop-artifact-bundle/v1"


def test_wave0_preset_names_expected_receipts():
    paths = preset_paths("wave0")
    assert "runs/studio_wave0/transfer_check.json" in paths
    assert "runs/mot/encode_device.json" in paths
