from __future__ import annotations

from substrate import v3io as io


def test_raw_verifier_file_snapshot_excludes_directories(tmp_path):
    (tmp_path / "receipt.json").write_text("{}")
    (tmp_path / "not-a-receipt.json").mkdir()

    assert io.regular_file_names(tmp_path) == {"receipt.json"}
    assert io.regular_file_names(tmp_path / "missing") == set()
