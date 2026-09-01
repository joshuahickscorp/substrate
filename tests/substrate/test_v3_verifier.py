from __future__ import annotations

from substrate import v3io as io


def test_raw_verifier_file_snapshot_excludes_directories(tmp_path):
    (tmp_path / "receipt.json").write_text("{}")
    (tmp_path / "not-a-receipt.json").mkdir()

    assert io.regular_file_names(tmp_path) == {"receipt.json"}
    assert io.regular_file_names(tmp_path / "missing") == set()


def test_source_path_snapshot_matches_recursive_python_glob(tmp_path):
    (tmp_path / "module.py").write_text("value = 1")
    (tmp_path / "package.py").mkdir()
    (tmp_path / "package.py" / "nested.py").write_text("value = 2")
    (tmp_path / "alias.py").symlink_to(tmp_path / "module.py")
    (tmp_path / "link.py").symlink_to(tmp_path / "package.py", target_is_directory=True)

    expected = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.py"))
    actual = sorted(path.relative_to(tmp_path) for path in io._source_paths((tmp_path,)))

    assert actual == expected
