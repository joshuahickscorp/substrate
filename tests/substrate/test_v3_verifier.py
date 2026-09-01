from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import asdict

from substrate import v3io as io
from substrate import v3principal as P
from substrate import v3verify as V


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


def test_compact_raw_report_projects_only_after_receipt_validation(tmp_path, monkeypatch):
    unit = P.work_units()[0]
    units = tmp_path / "units"
    checkpoints = tmp_path / "checkpoints"
    units.mkdir()
    checkpoints.mkdir()
    checkpoint = {"identity_hash": "checkpoint-identity"}
    receipt = {
        "schema": "substrate-v3-principal-unit/v1",
        "unit": asdict(unit),
        "cycles": [],
        "phases": [],
        "summary": {"episodes": 128, "checkpoint_exact": True, "body_continuity": True},
        "checkpoint": checkpoint,
        "activation": False,
    }
    receipt["receipt_identity"] = io.sha_obj(receipt)
    receipt["program"] = "substrate-v3"
    filename = f"{unit.identity}.json"
    (units / filename).write_text(json.dumps(receipt))
    (checkpoints / filename).write_text(json.dumps(checkpoint))
    monkeypatch.setattr(P, "UNITS", units)
    monkeypatch.setattr(P, "CHECKPOINTS", checkpoints)
    monkeypatch.setattr(P, "work_units", lambda: (unit,))

    report = V.raw(compact=True)

    assert report["all_pass"] is True
    assert report["mutation_checkpoint"] == checkpoint
    expected_unit = json.loads(json.dumps(asdict(unit)))
    assert report["receipts"][unit.identity] == {
        "activation": False,
        "cycles": [],
        "summary": receipt["summary"],
        "unit": expected_unit,
    }
    assert "checkpoint" not in report["receipts"][unit.identity]
    assert "phases" not in report["receipts"][unit.identity]


def test_paired_bootstrap_preserves_seeded_statistics():
    values = [0.1, 0.2, 0.3, 0.4]
    endpoint = "bootstrap-regression"
    seed = int(hashlib.sha256(endpoint.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    expected = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(2000)]

    actual = V.paired(values, endpoint)

    assert actual["bootstrap_95_ci"] == [
        V._percentile(expected, 0.025),
        V._percentile(expected, 0.975),
    ]
